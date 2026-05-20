"""Extended get_llm that adds LLM_PROVIDER=anthropic support to docquery."""

from __future__ import annotations

from docquery.config import Settings
from docquery.embeddings.llm import get_llm as _docquery_get_llm
from langchain_core.language_models import BaseChatModel


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    if settings is None:
        settings = Settings()
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=getattr(settings, "temperature", 0),
        )
    return _docquery_get_llm(settings)
