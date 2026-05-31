"""Tests for rosetta_ingest.node — no Ollama/Chroma required."""

from unittest.mock import MagicMock, patch

from rosetta_ingest.node import ingest_node


def _state(**kwargs):
    base = {"errors": [], "settings_dict": {}}
    base.update(kwargs)
    return base


def test_ingest_node_noop_missing_source_path():
    # No source_path → silent no-op; the generate workflow always pre-ingests.
    result = ingest_node(_state(db_path="/tmp/db"))
    assert result["errors"] == []


def test_ingest_node_missing_db_path():
    state = _state(source_path="/tmp/manual.pdf")
    result = ingest_node(state)
    assert any("db_path" in e for e in result["errors"])


def test_ingest_node_pdf_success(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("docquery.ingest", return_value=42) as mock_ingest:
        result = ingest_node(_state(source_path=str(pdf), db_path=str(tmp_path / "db")))

    assert result["errors"] == []
    mock_ingest.assert_called_once()
    items_arg = mock_ingest.call_args[0][0]
    assert str(pdf) in items_arg


def test_ingest_node_directory(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.c").write_text("int main() {}")
    (src / "util.h").write_text("void foo();")

    with patch("docquery.ingest", return_value=5) as mock_ingest:
        result = ingest_node(_state(source_path=str(src), db_path=str(tmp_path / "db")))

    assert result["errors"] == []
    items_arg = mock_ingest.call_args[0][0]
    assert any("main.c" in p for p in items_arg)
    assert any("util.h" in p for p in items_arg)


def test_ingest_node_empty_directory(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()

    with patch("docquery.ingest"):
        result = ingest_node(_state(source_path=str(src), db_path=str(tmp_path / "db")))

    assert any("no source files" in e for e in result["errors"])


def test_ingest_node_sets_db_path_on_settings(tmp_path):
    pdf = tmp_path / "m.pdf"
    pdf.write_bytes(b"%PDF")
    db = str(tmp_path / "mydb")
    captured = {}

    def fake_ingest(items, *, settings=None):
        captured["db_path"] = settings.db_path if settings else None
        return 1

    with patch("docquery.ingest", side_effect=fake_ingest):
        ingest_node(_state(source_path=str(pdf), db_path=db))

    assert captured["db_path"] == db


def test_ingest_node_noop_when_no_source_path(tmp_path):
    # No source_path → skip ingest silently; prior errors handled by LangGraph reducer.
    result = ingest_node(_state(source_path="", db_path="", errors=["prior error"]))
    assert result["errors"] == []


def test_ingest_node_handles_exception(tmp_path):
    pdf = tmp_path / "m.pdf"
    pdf.write_bytes(b"%PDF")

    with patch("docquery.ingest", side_effect=RuntimeError("boom")):
        result = ingest_node(_state(source_path=str(pdf), db_path=str(tmp_path / "db")))

    assert any("boom" in e for e in result["errors"])
