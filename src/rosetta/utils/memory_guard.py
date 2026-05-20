# Re-export shim — source of truth moved to rosetta-utils package.
from rosetta_utils.memory_guard import check_memory_headroom, log_memory  # noqa: F401

__all__ = ["check_memory_headroom", "log_memory"]
