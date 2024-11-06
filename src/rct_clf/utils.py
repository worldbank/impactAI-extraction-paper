from typing import Dict

from openai import AsyncOpenAI
from src.utils.openai_response import generate_openai_response_async


async def classify_rct(
    data: Dict,
    client: AsyncOpenAI,
    prompt_template: str,
    system_content: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict:
    """Process a single dictionary and extract metadata.

    Args:
        - data: Dictionary
        - client: OpenAI client
        - prompt_template: Prompt template
        - system_content: System content
        - model: Model to use
        - max_tokens: Maximum number of tokens in the response
        - temperature: Temperature for the response

    Returns:
        - Dict: JSON object
    """
    try:

        # Generate metadata
        render_dict = data
        rct = await generate_openai_response_async(
            client=client,
            render_dict=render_dict,
            prompt_template=prompt_template,
            system_content=system_content,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return rct
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
