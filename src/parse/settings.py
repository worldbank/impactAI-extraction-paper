from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDF2MarkdownSettings:
    """Settings for PDF to Markdown conversion."""

    path_input: Path = Path("data/raw/IDEAL")
    path_prompt: Path = Path("config/prompts/pdf2markdown-text-postprocess.prompt")
    path_prompt_tables: Path = Path(
        "config/prompts/pdf2markdown-table-postprocess.prompt"
    )
    path_examples: Path = Path("config/examples/parsing")
    save_metrics: bool = True
    verbose: bool = True
    batch_size: int = 1
    document_concurrency_limit: int = 10
    chunk_concurrency_limit: int = 20

    # Model settings
    model_text: str = None
    model_tables: str = None
    max_tokens: int = 16384

    @property
    def path_output(self) -> Path:
        return Path(
            f"data/processed/ATEST_docling_text_{self.model_text}_tables_{self.model_tables}"
        )

    @property
    def path_metrics(self) -> Path:
        return self.path_output / "parsing_metrics.json"
