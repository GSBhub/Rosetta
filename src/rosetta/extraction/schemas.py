# Re-export shim — source of truth moved to rosetta-schemas package.
from rosetta_schemas.models import ISAMeta, RegisterDef, InstructionDef, ISASpec  # noqa: F401

__all__ = ["ISAMeta", "RegisterDef", "InstructionDef", "ISASpec"]
