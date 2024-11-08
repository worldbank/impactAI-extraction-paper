import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.rct_clf.utils import compute_metrics, get_false_negatives
from src.utils.file_management import load_json_async, save_json_async
from src.rct_clf.settings import EvaluationParams


async def load_and_preprocess_data(
    path_preds: Path, path_true: Path
) -> Tuple[List[int], List[int]]:
    """
    Load and preprocess the prediction and ground truth data.

    Args:
        path_preds: Path to the predictions JSON file
        path_true: Path to the ground truth CSV file

    Returns:
        Tuple containing:
        - true_labels: List of true labels
        - pred_labels: List of predicted labels
    """
    logger = logging.getLogger(__name__)

    # Load data
    zsl_classified = await load_json_async(path_preds, logger)
    df_true = pd.read_csv(path_true)

    # Process predictions
    for key, value in zsl_classified.items():
        paper_id = int(key.split("_")[0].split("/")[-1])
        zsl_classified[key]["id"] = paper_id
        zsl_classified[key]["true_label"] = df_true.loc[
            df_true.PaperID == paper_id, "rct_encoded"
        ].values[0]

    # Extract labels
    true_labels = [
        int(zsl_classified[key]["true_label"]) for key in zsl_classified.keys()
    ]
    pred_labels = [
        1 if zsl_classified[key]["rct"] == "True" else 0
        for key in zsl_classified.keys()
    ]

    false_negatives = get_false_negatives(zsl_classified)

    return true_labels, pred_labels, false_negatives


async def main(path_preds: Path, path_true: Path, path_output: Path):
    """
    Main function to evaluate RCT classification.

    Args:
        path_preds: Path to predictions JSON file
        path_true: Path to ground truth CSV file
        path_output: Path to output metrics JSON file
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load and preprocess data
    true_labels, pred_labels, false_negatives = await load_and_preprocess_data(
        path_preds, path_true
    )

    # Compute metrics
    metrics = compute_metrics(true_labels, pred_labels)

    # Print results
    logger.info("Classification Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f"Precision: {metrics['precision']:.2f}%")
    logger.info(f"Recall: {metrics['recall']:.2f}%")
    logger.info(f"F1 Score: {metrics['f1']:.2f}%")

    if false_negatives:
        logger.info(f"\nNumber of False Negatives: {len(false_negatives)}")
        logger.info("False Negative cases:")
        for fn in false_negatives:
            logger.info(f"  - {fn}")

    # Save metrics
    await save_json_async(path_output, metrics, logger)


if __name__ == "__main__":
    import asyncio

    settings = EvaluationParams()
    asyncio.run(main(settings.path_preds, settings.path_true, settings.path_output))
