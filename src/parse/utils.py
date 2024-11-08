import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pymupdf4llm
import time
from tqdm import tqdm


from src.parse.settings import FitzSettings


def setup_logger(verbose: bool) -> logging.Logger:
    """
    Set up logger with appropriate verbosity.

    Args:
        verbose: If True, set level to INFO, else ERROR

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def get_pdf_files(input_path: Path, logger: logging.Logger) -> List[Path]:
    """
    Get list of PDF files from input directory.

    Args:
        input_path: Directory to search for PDFs
        logger: Logger instance

    Returns:
        List of PDF file paths
    """
    pdf_files = list(input_path.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_path}")
    else:
        logger.info(f"Found {len(pdf_files)} PDF files to process")
    return pdf_files


async def convert_pdf_to_markdown(
    pdf_path: Path,
    output_path: Path,
    logger: logging.Logger,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, Dict]:
    """
    Convert a single PDF file to markdown format asynchronously.

    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the markdown file
        logger: Logger instance
        semaphore: Semaphore for controlling concurrency

    Returns:
        Tuple containing:
        - str: Path to the output markdown file
        - Dict: Metrics about the conversion
    """
    async with semaphore:
        start_time = time.time()

        try:
            # Convert PDF to markdown
            loop = asyncio.get_event_loop()
            page_chunks = await loop.run_in_executor(
                None,
                lambda: pymupdf4llm.to_markdown(
                    str(pdf_path), page_chunks=True, show_progress=False
                ),
            )

            if not page_chunks:
                raise ValueError("No content extracted from PDF")

            # Extract text and metadata
            md_texts = [chunk.get("text", "") for chunk in page_chunks]
            md_text = "\n\n".join(md_texts)

            # Save markdown
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await loop.run_in_executor(None, lambda: output_path.write_text(md_text))

            # Basic metrics
            processing_time = time.time() - start_time
            return True, {"pages": len(page_chunks), "processing_time": processing_time}

        except Exception as e:
            logger.error(f"Error converting {pdf_path.name}: {str(e)}")
            return False, {"error": str(e)}


async def process_pdfs(
    pdf_files: List[Path], settings: "FitzSettings", logger: logging.Logger
) -> Dict:
    """
    Process multiple PDF files concurrently and collect metrics.

    Args:
        pdf_files: List of PDF files to process
        settings: FitzSettings instance
        logger: Logger instance

    Returns:
        Dictionary containing metrics for all processed files
    """
    semaphore = asyncio.Semaphore(1)  # Force sequential processing
    total_pages = 0
    total_time = 0
    successful = 0

    with tqdm(total=len(pdf_files), desc="Converting PDFs") as pbar:
        for pdf_path in pdf_files:
            relative_path = pdf_path.relative_to(settings.path_input)
            output_path = settings.path_output / relative_path.with_suffix(".md")

            success, metrics = await convert_pdf_to_markdown(
                pdf_path=pdf_path,
                output_path=output_path,
                logger=logger,
                semaphore=semaphore,
            )

            if success:
                total_pages += metrics["pages"]
                total_time += metrics["processing_time"]
                successful += 1

            pbar.update(1)
            await asyncio.sleep(0.1)  # Small delay between files

    return {
        "total_documents": len(pdf_files),
        "successful_conversions": successful,
        "total_pages": total_pages,
        "average_time_per_document": total_time / len(pdf_files) if pdf_files else 0,
        "average_time_per_page": total_time / total_pages if total_pages else 0,
    }


def log_summary_metrics(
    metrics: Dict, total_files: int, logger: logging.Logger
) -> None:
    """
    Calculate and log summary metrics.

    Args:
        metrics: Dictionary of metrics for all files
        total_files: Total number of files processed
        logger: Logger instance
    """
    logger.info("Conversion complete:")
    logger.info(
        f"Successfully converted: {metrics['successful_conversions']}/{metrics['total_documents']} files"
    )
    logger.info(f"Total pages processed: {metrics['total_pages']}")
    logger.info(
        f"Average time per document: {metrics['average_time_per_document']:.2f} seconds"
    )
    logger.info(
        f"Average time per page: {metrics['average_time_per_page']:.2f} seconds"
    )


async def save_metrics(
    metrics: Dict, metrics_path: Path, logger: logging.Logger
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics to save
        metrics_path: Path to save metrics
        logger: Logger instance
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: json.dump(metrics, open(metrics_path, "w"), indent=4)
    )
    logger.info(f"Metrics saved to {metrics_path}")
