import asyncio
import json
from pathlib import Path
from typing import List, Dict
import os
import aiofiles

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

from settings import Settings
from utils import load_prompt_async, process_pdf_async

load_dotenv()


async def process_files_async(
    files: List[Path],
    client: AsyncOpenAI,
    prompt_template: str,
    settings: Settings,
) -> Dict[str, dict]:
    """Process multiple PDF files asynchronously.

    Args:
        - files: List of paths to the PDF files
        - client: AsyncOpenAI client
        - prompt_template: Prompt template
        - settings: Settings

    Returns:
        - Dict[str, dict]: Dictionary of results with file paths as string keys
    """
    semaphore = asyncio.Semaphore(settings.batch_size)
    results = {}

    async def process_single_file(file: Path):
        """Process a single file with semaphore control."""
        async with semaphore:
            result = await process_pdf_async(
                filepath=file,
                client=client,
                prompt_template=prompt_template,
                model=settings.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )
            return str(file), result

    # Create and gather tasks
    tasks = [process_single_file(file) for file in files]

    # Process with progress bar
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing PDFs"
    ):
        file_path, result = await coro
        results[file_path] = result

    return results


async def main_async():
    """Main async function."""
    settings = Settings()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt_template = await load_prompt_async(settings.path_prompt)

    pdf_files = list(settings.path_folder.glob("*.pdf"))
    pdf_files.sort()

    if not pdf_files:
        print(f"No PDF files found in {settings.path_folder}")
        return

    results = await process_files_async(
        files=pdf_files,
        client=client,
        prompt_template=prompt_template,
        settings=settings,
    )

    output_path = settings.path_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(output_path, "w") as f:
        await f.write(json.dumps(results, indent=4, ensure_ascii=False))

    print(f"Results saved to {output_path}")


def main():
    """Entry point that runs the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
