import asyncio
from typing import Optional
from src.parse.settings import PDF2MarkdownSettings
from dotenv import load_dotenv
import argparse
from src.parse.utils import (
    setup_logger,
    get_pdf_files,
    main_process_pdfs,
    log_summary_metrics,
    save_metrics,
    setup_clients,
)

load_dotenv()


async def main(
    settings: Optional[PDF2MarkdownSettings] = None, n_samples: Optional[int] = None
) -> None:
    """
    Main function to convert PDFs to markdown.

    Args:
        use_gpt: Whether to use GPT capabilities
        use_docling: Whether to use Docling processing
        settings: FitzSettings or PDF2MarkdownSettings instance with configuration
        n_samples: Number of samples to process
    """
    # Setup
    logger = setup_logger(settings.verbose)

    # Get PDF files
    pdf_files = get_pdf_files(settings.path_input, logger)
    if not pdf_files:
        return

    if n_samples:
        pdf_files = [pdf for pdf in pdf_files[:n_samples] if pdf.stem.startswith("71.")]

    # Setup converter function and client if needed
    client_texts, client_tables = setup_clients(settings)
    logger.info("Using Docling processing")
    # Process PDFs concurrently
    metrics = await main_process_pdfs(
        pdf_files=pdf_files,
        settings=settings,
        logger=logger,
        client_texts=client_texts,
        client_tables=client_tables,
    )

    # Log summary
    log_summary_metrics(metrics, len(pdf_files), logger)

    # Save metrics if requested
    if settings.save_metrics:
        await save_metrics(metrics, settings.path_metrics, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert PDFs to markdown")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--n_samples", type=int, default=None, help="Number of samples to process"
    )

    args = parser.parse_args()

    settings_class = PDF2MarkdownSettings
    settings = settings_class(verbose=args.verbose)

    asyncio.run(main(settings=settings, n_samples=args.n_samples))
