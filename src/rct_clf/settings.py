from dataclasses import dataclass
from pathlib import Path


@dataclass
class ZSLSettings:
    path_prompt: Path = Path("config/prompts/RCT_ZSL.prompt")
    path_input: Path = Path("data/processed/metadata.json")
    path_output: Path = Path("data/processed/matadata_rct_classified.json")
    system_content: str = "You are a an expert in economic research."
    temperature: float = 1.0
    model: str = "gpt-4o"
    max_tokens: int = 1024
    batch_size: int = 10
