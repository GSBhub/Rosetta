# Re-export shim — source of truth moved to rosetta-mnemonics package.
from rosetta_mnemonics.discovery import (  # noqa: F401
    discover_mnemonics,
    _clean_mnemonic,
    _STRATEGIES,
)

__all__ = ["discover_mnemonics", "_clean_mnemonic", "_STRATEGIES"]
