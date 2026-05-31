"""Thin re-export of docquery's get_llm.

Anthropic support and Ollama timeout/num_predict are now in docquery upstream.
This module is kept so pcode_node and any other callers that import from here
continue to work without changes.
"""

from docquery.embeddings.llm import get_llm

__all__ = ["get_llm"]
