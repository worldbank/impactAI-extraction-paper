import asyncio
from src.parse.settings import FitzSettings
from src.parse.utils import (
    setup_logger,
    get_pdf_files,
    process_pdfs,
    log_summary_metrics,
    save_metrics,
)


async def main(settings: FitzSettings) -> None:
    """
    Main function to convert PDFs to markdown.

    Args:
        settings: FitzSettings instance with configuration
    """
    # Setup
    logger = setup_logger(settings.verbose)

    # Get PDF files
    pdf_files = get_pdf_files(settings.path_input, logger)
    if not pdf_files:
        return

    # Process PDFs concurrently
    metrics = await process_pdfs(pdf_files, settings, logger)

    # Log summary
    log_summary_metrics(metrics, len(pdf_files), logger)

    # Save metrics if requested
    if settings.save_metrics:
        await save_metrics(metrics, settings.path_metrics, logger)


if __name__ == "__main__":
    settings = FitzSettings()
    asyncio.run(main(settings))
