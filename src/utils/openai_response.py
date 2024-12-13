import json
import re
from jinja2 import Template
from openai import AsyncOpenAI
from google.generativeai import GenerativeModel
from typing import Dict, Optional, Tuple
from src.parse.settings import PDF2MarkdownSettings
from PIL import Image
import io
import base64
import logging


def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extract JSON from response using regex.

    Args:
        - response_text: Response text

    Returns:
        - Dict: JSON object
    """
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, response_text, re.DOTALL)

    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            return None
    return None


def extract_markdown_from_response(response_text: str) -> Optional[str]:
    """Extract markdown from response using regex.

    Args:
        - response_text: Response text

    Returns:
        - str: Markdown text
    """
    markdown_pattern = r"```markdown(.*?)```"
    matches = re.findall(markdown_pattern, response_text, re.DOTALL)

    if matches:
        return matches[0]

    return None


async def process_text_chunk(
    text: str,
    client: AsyncOpenAI,
    settings: PDF2MarkdownSettings,
    prompt_template: Template,
    logger: logging.Logger,
) -> Tuple[str, Dict[str, int]]:
    """
    Process a chunk of text using GPT/Gemini.

    Args:
        text: Text chunk to process
        client: API client
        settings: Settings instance
        prompt_template: Prompt template
        logger: Logger instance

    Returns:
        Tuple containing:
        - Processed text
        - Token metrics
    """
    try:
        response = await client.chat.completions.create(
            model=settings.model_text,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": text},
            ],
            max_tokens=settings.max_tokens,
            temperature=0.0,
        )

        try:
            result = extract_markdown_from_response(response.choices[0].message.content)
            metrics = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "failed_blocks": 0,
            }
        except Exception:
            result = "ERROR: Failed to process text chunk"
            metrics = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "failed_blocks": 1,
            }
        return result, metrics
    except Exception as e:
        logger.error(f"Error processing text chunk: {e}")
        raise


async def process_text_chunk_gemini(
    text: str,
    settings: PDF2MarkdownSettings,
    prompt_template: Template,
    logger: logging.Logger,
    client: GenerativeModel,
) -> Tuple[str, Dict[str, int]]:
    try:
        response = await client.generate_content_async(
            contents=[prompt_template, text],
            generation_config={
                "max_output_tokens": settings.max_tokens,
                "temperature": 0.0,
            },
        )

        try:
            result = extract_markdown_from_response(response.text)
            metrics = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
                "failed_blocks": 0,
            }
        except Exception:
            result = "ERROR: Failed to process text chunk"
            metrics = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "failed_blocks": 1,
            }
        return result, metrics
    except Exception as e:
        logger.error(f"Error processing text chunk: {e}")
        raise


async def generate_completion_with_image_docling(
    client: AsyncOpenAI,
    system_prompt: str,
    image_base64: str,
    max_tokens: int,
    temperature: float,
    user_markdown: str = None,
    model: str = "gpt-4o",
) -> Tuple[str, Dict[str, int]]:

    if user_markdown:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"#### Extracted Markdown:\n```markdown\n{user_markdown}\n```",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    }
                ],
            },
        ]

    # Make API call

    response = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    result_text = extract_markdown_from_response(response.choices[0].message.content)
    if not result_text:
        result_text = response.choices[0].message.content
    metrics = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return result_text, metrics


async def generate_completion_with_image_gemini(
    system_prompt: str,
    image_base64: str,
    max_tokens: int,
    temperature: float,
    user_markdown: str = None,
    client: GenerativeModel = None,
) -> Tuple[str, Dict[str, int]]:

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    if user_markdown:
        system_prompt += f"\n\nExtracted Markdown:\n```markdown\n{user_markdown}\n```"

    # Make API call

    response = await client.generate_content_async(
        contents=[system_prompt, image],
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )

    metrics = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count,
    }

    return response.text, metrics
