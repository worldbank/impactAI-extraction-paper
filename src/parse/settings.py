from dataclasses import dataclass
from pathlib import Path

from src.utils.file_management import get_secret


@dataclass
class PDF2MarkdownSettings:
    """Settings for PDF to Markdown conversion."""

    # GCP Bucket paths
    raw_bucket: str = get_secret("raw-bucket")
    processed_bucket: str = get_secret("processed-bucket")

    # Local paths
    path_input: Path = Path("/tmp/raw_pdfs")
    path_output: Path = Path("/tmp/processed_files")
    path_prompt: Path = Path("config/prompts/pdf2markdown-text-postprocess.prompt")
    path_metrics: Path = Path("/tmp/processed_files/metrics.json")

    path_prompt: Path = Path("config/prompts/pdf2markdown-text-postprocess.prompt")
    path_prompt_tables: Path = Path(
        "config/prompts/pdf2markdown-table-postprocess.prompt"
    )

    save_metrics: bool = True
    verbose: bool = True
    pdf_concurrency: int = 5
    chunk_concurrency: int = 30
    # Model settings
    model_text: str = "gpt-4o-mini"
    model_tables: str = "gpt-4o"
    max_tokens: int = 16384
