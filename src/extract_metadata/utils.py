import json
import re
from pathlib import Path
from typing import Dict, Optional

import pymupdf4llm
from jinja2 import Template
from openai import OpenAI


def load_prompt(path_prompt: Path) -> str:
    """Load prompt template from file.

    Args:
        - path_prompt: Path to the prompt template file

    Returns:
        - str: Prompt template
    """
    with open(path_prompt, "r") as f:
        return f.read()


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


def generate_metadata(
    client: OpenAI,
    md_text: str,
    prompt_template: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict:
    """Generate metadata using OpenAI API.

    Args:
        - client: OpenAI client
        - md_text: Markdown text
        - prompt_template: Prompt template
        - model: Model to use
        - max_tokens: Maximum number of tokens in the response
        - temperature: Temperature for the response

    Returns:
        - Dict: JSON object
    """
    template = Template(prompt_template)
    rendered_prompt = template.render(markdown_text=md_text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts metadata from academic papers.",
            },
            {"role": "user", "content": rendered_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = extract_json_from_response(response.choices[0].message.content)
    return result if result else {"error": "Failed to parse JSON response"}


def process_pdf(
    filepath: Path,
    client: OpenAI,
    prompt_template: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict:
    """Process a single PDF file and extract metadata.

    Args:
        - filepath: Path to the PDF file
        - client: OpenAI client
        - prompt_template: Prompt template
        - model: Model to use
        - max_tokens: Maximum number of tokens in the response
        - temperature: Temperature for the response

    Returns:
        - Dict: JSON object
    """
    try:
        # Try different page ranges until we get valid text

        md_text = None
        # error_messages = []

        md_text = pymupdf4llm.to_markdown(str(filepath), pages=[0, 1, 2])
        """
        for pages in page_ranges:
            try:
                md_text = pymupdf4llm.to_markdown(str(filepath), pages=pages)
                if md_text and len(md_text.strip()) > 100:  # Check if we got meaningful text
                    break
            except Exception as e:
                error_messages.append(f"Failed on pages {pages}: {str(e)}")

        if not md_text or len(md_text.strip()) <= 100:
            return {
                "filename": filepath.name,
                "error": f"Could not extract meaningful text from any page. Errors: {'; '.join(error_messages)}"
            }
        """
        # Generate metadata
        metadata = generate_metadata(
            client=client,
            md_text=md_text,
            prompt_template=prompt_template,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {"filename": filepath.name, "metadata": metadata}
    except Exception as e:
        return {"filename": filepath.name, "error": f"Unexpected error: {str(e)}"}
