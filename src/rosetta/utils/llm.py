# Re-export shim — source of truth moved to rosetta-utils package.
from rosetta_utils.llm import get_llm  # noqa: F401

__all__ = ["get_llm"]
