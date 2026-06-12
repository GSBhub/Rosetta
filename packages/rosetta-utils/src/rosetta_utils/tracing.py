"""LangSmith tracing helpers for Rosetta pipeline nodes.

Enable tracing by setting:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=rosetta   (optional, defaults to "default")

When these are not set, @traceable is a transparent no-op.
"""

from __future__ import annotations

from typing import Any


# Keys from PipelineState that are safe to log to LangSmith.
# Strips credentials, large binary blobs, and the live Chroma/settings objects.
_SAFE_STATE_KEYS = {
    "db_path", "processor_name", "out_dir", "source_path",
    "encoding_style", "output_format", "max_iterations", "max_instructions",
    "resume",
    "lang_dir", "compile_ok", "instruction_coverage", "register_overlap",
}


def state_summary(inputs: dict[str, Any]) -> dict[str, Any]:
    """process_inputs helper: strip sensitive/large fields before LangSmith logs them."""
    state = inputs.get("state", inputs)
    if not isinstance(state, dict):
        return inputs
    return {k: v for k, v in state.items() if k in _SAFE_STATE_KEYS}
