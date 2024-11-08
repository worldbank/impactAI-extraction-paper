from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    path_folder: Path = Path("data/raw")
    path_prompt: Path = Path("config/prompts/metadata-extraction.prompt")
    path_output: Path = Path("data/processed/metadata.json")
    system_content: str = (
        "You are a helpful assistant that extracts metadata from academic papers."
    )
    temperature: float = 0.0
    model: str = "gpt-4o"
    max_tokens: int = 1024
    batch_size: int = 10
