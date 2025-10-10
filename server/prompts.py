"""
Bilingual Prompt Templates
English and Japanese prompts for VLM models
"""

from typing import Dict

# English prompts (original)
PROMPTS_EN: Dict[str, str] = {
    "caption": "Describe briefly.",
    "detail": "What's happening?",
    "action": "Main action?",
    "objects": "List objects.",
    "query_default": "Describe this image.",
}

# Japanese prompts (for translation to English before VLM)
# These will be translated to English by the translator
PROMPTS_JA: Dict[str, str] = {
    "caption": "簡潔に説明してください。",
    "detail": "何が起きていますか？",
    "action": "主なアクションは何ですか？",
    "objects": "オブジェクトをリストしてください。",
    "query_default": "この画像を説明してください。",
}

# Moondream-specific prompts
MOONDREAM_PROMPTS_EN: Dict[str, str] = {
    "caption": "Describe this image in detail.",
    "query": "Answer the question based on the image.",
    "detect": "Detect and list all objects in this image.",
    "point": "Point out the location of objects in this image.",
}

MOONDREAM_PROMPTS_JA: Dict[str, str] = {
    "caption": "この画像を詳細に説明してください。",
    "query": "画像に基づいて質問に答えてください。",
    "detect": "この画像内のすべてのオブジェクトを検出してリストしてください。",
    "point": "この画像内のオブジェクトの位置を指摘してください。",
}


def get_prompt(
    prompt_type: str,
    language: str = "en",
    model: str = "smolvlm",
    custom_query: str = None
) -> str:
    """
    Get appropriate prompt based on language and model

    Args:
        prompt_type: Type of prompt ("caption", "detail", "action", "objects", "query")
        language: Language code ("en" or "ja")
        model: Model name ("smolvlm" or "moondream")
        custom_query: Custom user query (overrides prompt_type)

    Returns:
        Prompt string
    """
    # If custom query provided, use it directly
    if custom_query:
        return custom_query

    # Select prompt set based on model and language
    if model == "moondream":
        prompts = MOONDREAM_PROMPTS_JA if language == "ja" else MOONDREAM_PROMPTS_EN
    else:
        prompts = PROMPTS_JA if language == "ja" else PROMPTS_EN

    # Return prompt or default
    return prompts.get(prompt_type, prompts.get("query_default", "Describe this image."))
