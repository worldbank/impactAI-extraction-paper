import logging
from pathlib import Path

import aiofiles
import json
from typing import Dict, List


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
