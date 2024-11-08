import asyncio
from typing import Dict
import os
import logging

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

from settings import ZSLSettings
from utils import classify_rct
from src.utils.file_management import (
    load_prompt_async,
    save_json_async,
    load_json_async,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def process_data_async(
    data: Dict,
    client: AsyncOpenAI,
    prompt_template: str,
    system_content: str,
    settings: ZSLSettings,
) -> Dict[str, dict]:
    """Process multiple dictionaries asynchronously.

    Args:
        - data: Dictionary of data to process
        - client: AsyncOpenAI client
        - prompt_template: Prompt template
        - system_content: System content for the prompt
        - settings: Settings

    Returns:
        - Dict[str, dict]: Dictionary of results with file paths as string keys
    """
    semaphore = asyncio.Semaphore(settings.batch_size)
    results = {}

    async def process_single_data(key: str, metadata: Dict):
        """Process a single data entry with semaphore control."""
        async with semaphore:
            result = await classify_rct(
                data=metadata,
                client=client,
                prompt_template=prompt_template,
                system_content=system_content,
                model=settings.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )
            return key, result

    # Create tasks list
    tasks = []
    for key, value in data.items():
        if metadata := value.get("metadata"):
            tasks.append(process_single_data(key, metadata))

    # Process with progress bar
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing data"
    ):
        key, result = await coro
        results[key] = {**data[key], **result}

    return results


async def main_async():
    """Main async function."""
    settings = ZSLSettings()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt_template = await load_prompt_async(settings.path_prompt, logger)

    data = await load_json_async(settings.path_input, logger)

    if not data:
        logging.warning(f"No data found in {settings.path_input}")
        return

    results = await process_data_async(
        data=data,
        client=client,
        prompt_template=prompt_template,
        system_content=settings.system_content,
        settings=settings,
    )

    output_path = settings.path_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    await save_json_async(output_path, results, logger)


def main():
    """Entry point that runs the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
