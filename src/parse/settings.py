from dataclasses import dataclass
from pathlib import Path


@dataclass
class FitzSettings:
    """Settings for PDF parsing with PyMuPDF."""

    path_input: Path = Path("data/raw")
    path_output: Path = Path("data/processed/fitz_markdown")
    save_metrics: bool = True
    path_metrics: Path = Path("data/processed/fitz_markdown/parsing_metrics.json")
    verbose: bool = False
