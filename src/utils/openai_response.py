import json
import re
from jinja2 import Template
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Optional


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_openai_response_async(
    client: AsyncOpenAI,
    render_dict: Dict,
    system_content: str,
    prompt_template: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict:
    """Generate metadata using OpenAI API.

    Args:
        - client: OpenAI client
        - render_dict: Dictionary to render
        - prompt_template: Prompt template
        - system_content: System content
        - model: Model to use
        - max_tokens: Maximum number of tokens in the response
        - temperature: Temperature for the response

    Returns:
        - Dict: JSON object
    """
    template = Template(prompt_template)

    if render_dict.get("keywords") == "Not Available":
        keywords = None
    else:
        keywords = render_dict.get("keywords")

    rendered_prompt = template.render(
        abstract=render_dict.get("abstract"),
        title=render_dict.get("title"),
        keywords=keywords,
        markdown_text=render_dict.get("markdown_text"),
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": rendered_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = extract_json_from_response(response.choices[0].message.content)
    return result if result else {"error": "Failed to parse JSON response"}
