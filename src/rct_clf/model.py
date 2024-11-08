import logging
import torch
from os.path import join

import numpy as np
from dataclasses import asdict
from setfit import SetFitModel, TrainingArguments, Trainer

from src.rct_clf.settings import TrainingParams
from src.rct_clf.utils import compute_metrics


class RCTModel:
    """
    A class used to represent a RCTModel

    Attributes
    ----------
    path_config:
        a formatted string to load fine-tuning configuration
    """

    def __init__(self, path_models, output_model_name):
        self.path_models = path_models
        self.config = self._load_config(self.path_models, output_model_name)
        self.logger: logging.Logger = logging.getLogger(__name__)

    def load_model(self):
        """
        Loads a transformer model from a local path or the HuggingFace repository.

        Returns:
            - model: SetFitModel, the loaded model
        """

        self.logger.info("Loading untrained model from HuggingFace repository")
        model = SetFitModel.from_pretrained(self.config.get("model_name"))

        return model

    def _load_config(self, path_models, output_model_name):
        """
        Loads the configuration for the model.

        Args:
            - path_models: Path to the models directory
            - output_model_name: Name of the output model file

        Returns:
            - config: Dict, the configuration for the model
        """
        config = TrainingParams()
        config = asdict(config)
        config["output_dir"] = join(path_models, output_model_name)

        return config

    def upload_model(self, model):
        """
        Saves a transformer model to a local path and/or the HuggingFace repository.

        Args:
            - model: SetFitModel, the model to save

        Returns:
            - None
        """
        model_path = self.config["output_dir"]
        model.save_model(model_path)

        # Push to the Hub when `save_model` is called by the user.
        if self.config.get("push_to_hub"):
            self.logger.info("Pushing model to the HuggingFace Hub")
            model.push_to_hub(commit_message="Model save")

    def train(self, train_dataset, eval_dataset, model):
        """
        Trains a model on a given dataset.

        Args:
            - train_dataset: Dataset, the dataset to train the model on
            - eval_dataset: Dataset, the dataset to evaluate the model on
            - model: SetFitModel, the model to train

        Returns:
            - trainer: Trainer, the trained model
            - eval_dict: Dict, the evaluation dictionary
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")

        # Move model to appropriate device
        model = model.to(device)

        args = TrainingArguments.from_dict(arguments=self.config, ignore_extra=True)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric="accuracy",
            column_mapping={"abstract": "text", "label": "label"},
        )

        trainer.train()
        eval_dict = trainer.evaluate()
        return trainer, eval_dict

    def predict(self, model, dataset):
        """
        Makes predictions on a given dataset using a given model.

        Args:
            - model: SetFitModel, the model to use for making predictions
            - dataset: Dataset, the dataset to make predictions on

        Returns:
            - y_preds: List, the predictions
        """
        y_preds = model(dataset["text"])

        return y_preds

    def evaluate(self, y_true, model, dataset, variable_name=None):
        """
        Evaluates a model on a given dataset.

        Args:
            - y_true: List, the true labels
            - model: SetFitModel, the model to evaluate
            - dataset: Dataset, the dataset to evaluate the model on
            - variable_name: Str, the name of the variable to use for stratified evaluation

        Returns:
            - eval_dict: Dict, the evaluation dictionary
        """
        y_true = np.array(y_true)
        y_pred = self.predict(model, dataset)
        y_pred = y_pred.cpu().numpy()
        eval_dict = compute_metrics(y_true, y_pred)
        if variable_name:
            assert (
                variable_name in dataset.column_names
            ), f"{variable_name} not in dataset"
            unique_variable_labels = dataset.unique(variable_name)
            unique_variable_labels.sort()
            for label in unique_variable_labels:
                indices = dataset.map(
                    lambda example, idx: (
                        {"index": idx}
                        if example[variable_name] == label
                        else {"index": None}
                    ),
                    with_indices=True,
                    remove_columns=dataset.column_names,
                )
                indices = [
                    idx.get("index") for idx in indices if idx.get("index", None)
                ]
                eval_dict[label] = compute_metrics(y_true[indices], y_pred[indices])
        return eval_dict
