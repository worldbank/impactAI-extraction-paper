import logging
from pathlib import Path

import aiofiles
import json
from typing import Dict, List

from google.cloud import storage, secretmanager
from tqdm import tqdm
from google.auth import default


async def load_prompt_async(path_prompt: Path, logger: logging.Logger) -> str:
    """Load prompt template from file.

    Args:
        - path_prompt: Path to the prompt template file

    Returns:
        - str: Prompt template
    """
    async with aiofiles.open(path_prompt, "r") as f:
        logger.info(f"Loading prompt from {path_prompt}")
        return await f.read()


async def load_examples(path_examples: Path, logger: logging.Logger) -> List[Dict]:
    """Load examples from file."""
    examples = []
    for path_example in path_examples.glob("*.json"):
        examples.append(await load_json_async(path_example, logger))

    return examples


async def load_json_async(path_json: Path, logger: logging.Logger) -> Dict:
    """Load JSON from file.

    Args:
        - path_json: Path to the JSON file

    Returns:
        - Dict: JSON object
    """
    async with aiofiles.open(path_json, "r") as f:
        logger.info(f"Loading JSON from {path_json}")
        return json.loads(await f.read())


async def save_json_async(path_json: Path, data: Dict, logger: logging.Logger) -> None:
    """Save JSON to file.

    Args:
        - path_json: Path to the JSON file
        - data: Data to save
    """
    async with aiofiles.open(path_json, "w") as f:
        await f.write(json.dumps(data, indent=4, ensure_ascii=False))
        logger.info(f"Results saved to {path_json}")


def get_secret(secret_id: str) -> str:
    """
    Fetch a secret from GCP Secret Manager

    Args:
        secret_id: The ID of the secret to fetch

    Returns:
        The secret value
    """
    credentials, project = default()
    client = secretmanager.SecretManagerServiceClient(credentials=credentials)
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


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


def download_pdfs_from_bucket(
    bucket_name: str,
    local_folder: str | Path,
    logger: logging.Logger,
    already_processed_pdfs: List[str],
) -> List[Path]:
    """
    Download PDF files from a GCP bucket to a local folder.

    Args:
        bucket_name: Name of the GCP bucket.
        local_folder: Local folder to store downloaded files (str or Path).
        logger: Logger instance.

    Returns:
        List of downloaded file paths.
    """
    # Convert string path to Path object if necessary
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    downloaded_files = []
    for blob in blobs:
        if (
            blob.name.endswith(".pdf") and blob.name not in already_processed_pdfs
        ):  # Ensure only PDFs are downloaded
            local_path = local_folder / blob.name
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {blob.name} to {local_path}")
            downloaded_files.append(local_path)

    if not downloaded_files:
        logger.warning(f"No PDFs found in bucket {bucket_name}")
    return downloaded_files


def upload_to_bucket(
    bucket_name: str, local_folder: Path, logger: logging.Logger
) -> None:
    """
    Upload processed files from a local folder to a GCP bucket.

    Args:
        bucket_name: Name of the GCP bucket.
        local_folder: Folder containing files to upload.
        logger: Logger instance.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for file_path in tqdm(local_folder.glob("*"), desc="Uploading processed files"):
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(str(file_path))
        logger.info(f"Uploaded {file_path.name} to bucket {bucket_name}")


def download_processed_list_and_metrics(
    bucket_name: str, local_folder: str | Path, logger: logging.Logger
) -> tuple[List[str], Dict]:
    """
    Download processed PDFs list and metrics from a GCP bucket.

    Args:
        bucket_name: Name of the GCP bucket.
        local_folder: Local folder to store downloaded files (str or Path).
        logger: Logger instance.

    Returns:
        Tuple containing:
        - List of processed PDF names
        - Dictionary containing metrics data
    """
    # Convert string path to Path object if necessary
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    processed_pdfs = []
    metrics_data = {}

    # Download and read processed_pdfs.txt
    processed_list_blob = bucket.blob("processed_pdfs.txt")
    if processed_list_blob.exists():
        local_list_path = local_folder / "processed_pdfs.txt"
        processed_list_blob.download_to_filename(local_list_path)
        logger.info(f"Downloaded processed PDFs list to {local_list_path}")
        with open(local_list_path, "r") as f:
            processed_pdfs = [line.strip() for line in f.readlines()]
    else:
        logger.warning("No processed_pdfs.txt found in bucket")

    # Download and read metrics.json
    metrics_blob = bucket.blob("metrics.json")
    if metrics_blob.exists():
        local_metrics_path = local_folder / "metrics.json"
        metrics_blob.download_to_filename(local_metrics_path)
        logger.info(f"Downloaded metrics to {local_metrics_path}")
        with open(local_metrics_path, "r") as f:
            metrics_data = json.load(f)
    else:
        logger.warning("No metrics.json found in bucket")

    return processed_pdfs, metrics_data
