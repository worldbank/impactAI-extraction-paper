from pathlib import Path
from typing import Dict

import pymupdf4llm
from openai import AsyncOpenAI
from src.utils.openai_response import generate_openai_response_async


async def process_pdf_async(
    filepath: Path,
    client: AsyncOpenAI,
    prompt_template: str,
    system_content: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict:
    """Process a single PDF file and extract metadata.

    Args:
        - filepath: Path to the PDF file
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
        # Try different page ranges until we get valid text

        md_text = None
        page_ranges = [[0, 1, 2], [0], [1], [2]]
        error_messages = []

        for pages in page_ranges:
            try:
                md_text = pymupdf4llm.to_markdown(str(filepath), pages=pages)
                if (
                    md_text and len(md_text.strip()) > 100
                ):  # Check if we got meaningful text
                    break
            except Exception as e:
                error_messages.append(f"Failed on pages {pages}: {str(e)}")

        if not md_text or len(md_text.strip()) <= 100:
            return {
                "filename": filepath.name,
                "error": f"Could not extract meaningful text from any page. Errors: {'; '.join(error_messages)}",
            }

        # Generate metadata
        render_dict = {"markdown_text": md_text}
        metadata = await generate_openai_response_async(
            client=client,
            render_dict=render_dict,
            prompt_template=prompt_template,
            system_content=system_content,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {"filename": filepath.name, "metadata": metadata}
    except Exception as e:
        return {"filename": filepath.name, "error": f"Unexpected error: {str(e)}"}
