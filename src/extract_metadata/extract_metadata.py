import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

from settings import Settings
from utils import load_prompt, process_pdf
import os

load_dotenv()


def process_files_parallel(
    files: List[Path], client: OpenAI, prompt_template: str, settings: Settings
) -> Dict[str, dict]:
    """Process multiple PDF files in parallel.

    Args:
        - files: List of paths to the PDF files
        - client: OpenAI client
        - prompt_template: Prompt template
        - settings: Settings

    Returns:
        - Dict[str, dict]: Dictionary of results with file paths as string keys
    """
    results = {}

    with ThreadPoolExecutor(max_workers=settings.batch_size) as executor:
        # Create a list of future tasks
        futures = {
            str(file): executor.submit(  # Convert Path to string for JSON serialization
                process_pdf,
                filepath=file,
                client=client,
                prompt_template=prompt_template,
                model=settings.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )
            for file in files
        }

        # Process results as they complete with progress bar
        for key, future in tqdm(futures.items(), desc="Processing PDFs"):
            results[key] = future.result()

    return results


def main():
    settings = Settings()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt_template = load_prompt(settings.path_prompt)

    pdf_files = list(settings.path_folder.glob("*.pdf"))
    pdf_files.sort()

    if not pdf_files:
        print(f"No PDF files found in {settings.path_folder}")
        return

    results = process_files_parallel(
        files=pdf_files,
        client=client,
        prompt_template=prompt_template,
        settings=settings,
    )

    output_path = settings.path_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
