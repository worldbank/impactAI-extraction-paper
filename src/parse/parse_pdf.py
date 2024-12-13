import asyncio
from typing import Optional
from src.parse.settings import PDF2MarkdownSettings
from dotenv import load_dotenv
import argparse
from src.parse.utils import (
    setup_logger,
    main_process_pdfs,
    log_summary_metrics,
    save_metrics,
    setup_clients,
)

from src.utils.file_management import download_pdfs_from_bucket, upload_to_bucket
import os

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

    # Create temporary directory if it doesn't exist
    temp_dir = "/tmp/pdf_processing"
    os.makedirs(temp_dir, exist_ok=True)

    # Update settings path_input to use temp directory
    settings.path_input = temp_dir

    # Get PDF files
    pdf_files = download_pdfs_from_bucket(
        settings.raw_bucket, settings.path_input, logger
    )

    if not pdf_files:
        logger.error("No PDFs found in bucket")
        return

    if n_samples:
        pdf_files = [pdf for pdf in pdf_files[:n_samples]]

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
    logger.info("Uploading processed files to GCP bucket...")
    upload_to_bucket(settings.processed_bucket, settings.path_output, logger)
    # Save metrics if requested
    if settings.save_metrics:
        await save_metrics(metrics, settings.path_metrics, logger)

    # Clean up temporary files after processing
    try:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Error cleaning up temporary directory: {e}")


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
