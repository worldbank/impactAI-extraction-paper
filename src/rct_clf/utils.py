from typing import Dict, List

from openai import AsyncOpenAI
from src.utils.openai_response import generate_openai_response_async
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def get_false_negatives(zsl_classified: dict) -> List[str]:
    """
    Get list of false negative cases.

    Args:
        zsl_classified: Dictionary containing classifications

    Returns:
        List of keys corresponding to false negative cases
    """
    return [
        key
        for key in zsl_classified.keys()
        if zsl_classified[key]["true_label"] == 1
        and zsl_classified[key]["rct"] == "False"
    ]


def compute_metrics(true_labels: List[int], pred_labels: List[int]) -> dict:
    """
    Compute classification metrics.

    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels

    Returns:
        Dictionary containing the metrics
    """
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels) * 100,
        "precision": precision_score(true_labels, pred_labels) * 100,
        "recall": recall_score(true_labels, pred_labels) * 100,
        "f1": f1_score(true_labels, pred_labels) * 100,
    }

    return metrics
