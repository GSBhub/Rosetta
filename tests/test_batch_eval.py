"""Tests for manifest loading and environment variable resolution."""

import os
from pathlib import Path

import pytest
import yaml

from rosetta.evaluation.batch_eval import _resolve_env, load_manifest


def test_resolve_env_no_vars():
    assert _resolve_env("hello world") == "hello world"


def test_resolve_env_substitution(monkeypatch):
    monkeypatch.setenv("MY_TEST_VAR", "/some/path")
    assert _resolve_env("$MY_TEST_VAR/sub") == "/some/path/sub"


def test_resolve_env_missing_var():
    # Unset var should remain as literal text
    os.environ.pop("NONEXISTENT_VAR_XYZ", None)
    result = _resolve_env("$NONEXISTENT_VAR_XYZ/path")
    assert result == "$NONEXISTENT_VAR_XYZ/path"


def test_load_manifest_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("GHIDRA_HOME", "/fake/ghidra")
    manifest_data = {
        "targets": [
            {
                "id": "armv7",
                "name": "ARM_v7",
                "manual": "manuals/arm.pdf",
                "db": "dbs/armv7.db",
                "ghidra_reference_lang": "ARM:LE:32:v7",
                "reference_slaspec": "$GHIDRA_HOME/Ghidra/Processors/ARM/data/languages/ARM7_le.slaspec",
            }
        ]
    }
    manifest_path = tmp_path / "test.yaml"
    manifest_path.write_text(yaml.dump(manifest_data))

    targets = load_manifest(manifest_path)
    assert len(targets) == 1
    assert targets[0]["id"] == "armv7"
    # Env var should be resolved
    assert "/fake/ghidra" in targets[0]["reference_slaspec"]


def test_load_manifest_multiple_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("GHIDRA_HOME", "/g")
    manifest_data = {
        "targets": [
            {"id": f"arm{i}", "name": f"ARM_{i}", "manual": f"m{i}.pdf",
             "db": f"d{i}.db", "ghidra_reference_lang": "ARM:LE:32:v7",
             "reference_slaspec": "$GHIDRA_HOME/fake.slaspec"}
            for i in range(4)
        ]
    }
    manifest_path = tmp_path / "multi.yaml"
    manifest_path.write_text(yaml.dump(manifest_data))

    targets = load_manifest(manifest_path)
    assert len(targets) == 4
    assert {t["id"] for t in targets} == {"arm0", "arm1", "arm2", "arm3"}


def test_load_manifest_empty(tmp_path):
    manifest_path = tmp_path / "empty.yaml"
    manifest_path.write_text("targets: []")
    targets = load_manifest(manifest_path)
    assert targets == []
