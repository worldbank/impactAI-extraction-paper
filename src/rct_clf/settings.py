from dataclasses import dataclass
from pathlib import Path


@dataclass
class ZSLSettings:
    path_prompt: Path = Path("config/prompts/RCT_ZSL.prompt")
    path_input: Path = Path("data/processed/metadata.json")
    path_output: Path = Path("data/processed/matadata_rct_classified.json")
    system_content: str = "You are a an expert in economic research and in classifiying studies as Randomized Controlled Trial (RCT) or not."
    temperature: float = 1.0
    model: str = "gpt-4o"
    max_tokens: int = 1024
    batch_size: int = 10


@dataclass
class TrainingParams:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 32
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    output_dir: str = "SetFitModel"
    num_iteration: int = 3


@dataclass
class EvaluationParams:
    path_preds: Path = Path("data/processed/matadata_rct_classified.json")
    path_true: Path = Path("data/raw/RCT_GT.csv")
    path_output: Path = Path("data/processed/ZSL_two_steps_metrics.json")
