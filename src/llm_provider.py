import ollama

from config import get_ollama_base_url, get_llm_provider, get_zhipu_api_key, get_zhipu_model

_selected_model: str | None = None


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=get_ollama_base_url())


def list_models() -> list[str]:
    """
    Lists all models available on the local Ollama server.

    Returns:
        models (list[str]): Sorted list of model names.
    """
    response = _ollama_client().list()
    return sorted(m.model for m in response.models)


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
    Generates text using the configured LLM provider (GLM or Ollama).

    Args:
        prompt (str): User prompt
        model_name (str): Optional model name override

    Returns:
        response (str): Generated text
    """
    provider = get_llm_provider()

    if provider == "glm":
        from zhipuai import ZhipuAI
        api_key = get_zhipu_api_key()
        if not api_key:
            raise RuntimeError("zhipu_api_key is not set in config.json")
        model = model_name or get_zhipu_model() or "glm-4-flash"
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

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
