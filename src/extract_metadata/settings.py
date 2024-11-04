from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    path_folder: Path = Path("data/raw")
    path_prompt: Path = Path("config/prompts/metadata-extraction.prompt")
    path_output: Path = Path("data/processed/metadata.json")
    temperature: float = 0.0
    model: str = "gpt-4"
    max_tokens: int = 1024
    batch_size: int = 10
