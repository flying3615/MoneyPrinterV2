import ollama
from openai import OpenAI

from config import (
    get_ollama_base_url,
    get_llm_provider,
    get_openai_api_key,
    get_openai_base_url,
    get_openai_model,
)

_selected_model: str | None = None


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=get_ollama_base_url())


def _openai_client() -> OpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "openai_api_key is not set in config.json and OPENAI_API_KEY is not set"
        )
    base_url = get_openai_base_url() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def list_models() -> list[str]:
    """
    Lists available models for the configured text provider.

    Returns:
        models (list[str]): Sorted list of model names.
    """
    provider = get_llm_provider()
    if provider == "ollama":
        response = _ollama_client().list()
        return sorted(m.model for m in response.models)

    configured = get_openai_model().strip()
    return [configured] if configured else []


def select_model(model: str) -> None:
    """
    Sets the model to use for all subsequent generate_text calls.

    Args:
        model (str): An Ollama model name (must be already pulled).
    """
    global _selected_model
    _selected_model = model


def get_active_model() -> str | None:
    """
    Returns the currently selected model, or None if none has been selected.
    """
    return _selected_model


def generate_text(prompt: str, model_name: str = None) -> str:
    """
    Generates text using the configured LLM provider.

    Args:
        prompt (str): User prompt
        model_name (str): Optional model name override

    Returns:
        response (str): Generated text
    """
    provider = get_llm_provider()

    if provider != "ollama":
        model = model_name or _selected_model or get_openai_model()
        if not model:
            raise RuntimeError(
                "No OpenAI-compatible model configured. Set openai_model or OPENAI_MODEL."
            )
        client = _openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning_split": True},
        )
        return (response.choices[0].message.content or "").strip()

    # Default: Ollama
    model = model_name or _selected_model
    if not model:
        raise RuntimeError(
            "No Ollama model selected. Call select_model() first or pass model_name."
        )
    response = _ollama_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()
