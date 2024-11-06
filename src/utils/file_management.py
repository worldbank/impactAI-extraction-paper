from pathlib import Path
import aiofiles


async def load_prompt_async(path_prompt: Path) -> str:
    """Load prompt template from file.

    Args:
        - path_prompt: Path to the prompt template file

    Returns:
        - str: Prompt template
    """
    async with aiofiles.open(path_prompt, "r") as f:
        return await f.read()
