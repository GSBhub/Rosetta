"""Tests for rosetta_mnemonics.discovery — no Ollama required."""

from unittest.mock import MagicMock, patch

from rosetta_mnemonics.discovery import _clean_mnemonic, discover_mnemonics, _MnemonicList, _InstructionCount


# ---------------------------------------------------------------------------
# _clean_mnemonic
# ---------------------------------------------------------------------------

def test_clean_mnemonic_uppercase():
    assert _clean_mnemonic("add") == "ADD"


def test_clean_mnemonic_valid_with_suffix():
    assert _clean_mnemonic("VADD.F32") == "VADD.F32"


def test_clean_mnemonic_rejects_spaces():
    assert _clean_mnemonic("ADD R0") == ""


def test_clean_mnemonic_rejects_commas():
    assert _clean_mnemonic("ADD,SUB") == ""


def test_clean_mnemonic_rejects_parens():
    assert _clean_mnemonic("ADD(R0)") == ""


def test_clean_mnemonic_empty():
    assert _clean_mnemonic("") == ""


# ---------------------------------------------------------------------------
# discover_mnemonics
# ---------------------------------------------------------------------------

def _make_pipeline_mock(mnemonics_sequence):
    """Return an ExtractionPipeline class mock that yields results from a sequence."""
    call_count = [0]
    instances = []

    class FakePipeline:
        def __init__(self, **kwargs):
            instances.append(self)

        def run(self, query):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(mnemonics_sequence):
                item = mnemonics_sequence[idx]
                if isinstance(item, int):
                    return _InstructionCount(count=item)
                return _MnemonicList(mnemonics=item)
            return _MnemonicList(mnemonics=[])

    return FakePipeline


def test_discover_mnemonics_deduplicates():
    # count=3, strategy 0 returns duplicates, strategy 1 returns overlap
    pipeline_cls = _make_pipeline_mock([3, ["ADD", "ADD", "SUB"], ["SUB", "MOV"]])
    settings = MagicMock()
    settings.db_path = "/tmp/db"
    settings.vs = MagicMock()

    with patch("rosetta_mnemonics.discovery.ExtractionPipeline", pipeline_cls), \
         patch("rosetta_mnemonics.discovery._build_chroma"):
        result = discover_mnemonics("/tmp/db", settings, strategies=["s0", "s1"])

    assert sorted(result) == ["ADD", "MOV", "SUB"]


def test_discover_mnemonics_uppercases():
    pipeline_cls = _make_pipeline_mock([2, ["add", "sub"]])
    settings = MagicMock()
    settings.vs = MagicMock()

    with patch("rosetta_mnemonics.discovery.ExtractionPipeline", pipeline_cls), \
         patch("rosetta_mnemonics.discovery._build_chroma"):
        result = discover_mnemonics("/tmp/db", settings, strategies=["s0"])

    assert "ADD" in result
    assert "SUB" in result


def test_discover_mnemonics_filters_invalid():
    pipeline_cls = _make_pipeline_mock([5, ["ADD", "invalid mnemonic", "SUB", "ADD,DUP"]])
    settings = MagicMock()
    settings.vs = MagicMock()

    with patch("rosetta_mnemonics.discovery.ExtractionPipeline", pipeline_cls), \
         patch("rosetta_mnemonics.discovery._build_chroma"):
        result = discover_mnemonics("/tmp/db", settings, strategies=["s0"])

    assert "ADD" in result
    assert "SUB" in result
    assert not any(" " in m or "," in m for m in result)


def test_discover_mnemonics_handles_strategy_failure():
    call_count = [0]

    class BrokenPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, query):
            call_count[0] += 1
            if call_count[0] == 1:
                return _InstructionCount(count=2)
            raise RuntimeError("network error")

    settings = MagicMock()
    settings.vs = MagicMock()

    with patch("rosetta_mnemonics.discovery.ExtractionPipeline", BrokenPipeline), \
         patch("rosetta_mnemonics.discovery._build_chroma"):
        result = discover_mnemonics("/tmp/db", settings, strategies=["s0"])

    assert result == []


def test_discover_mnemonics_sets_db_path():
    settings = MagicMock()
    settings.vs = MagicMock()

    pipeline_cls = _make_pipeline_mock([0])
    with patch("rosetta_mnemonics.discovery.ExtractionPipeline", pipeline_cls), \
         patch("rosetta_mnemonics.discovery._build_chroma") as mock_build:
        discover_mnemonics("/special/db", settings, strategies=[])

    assert settings.db_path == "/special/db"
    mock_build.assert_called_once_with(settings)
