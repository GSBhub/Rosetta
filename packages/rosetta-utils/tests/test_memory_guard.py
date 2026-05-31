"""Tests for memory_guard — no LLM or external services required."""

from unittest.mock import patch, MagicMock
import pytest
from rosetta_utils.memory_guard import check_memory_headroom, log_memory


def test_log_memory_runs(caplog):
    with patch("psutil.Process") as mock_proc:
        mock_proc.return_value.memory_info.return_value.rss = 2 * 1024**3
        import logging
        with caplog.at_level(logging.INFO, logger="rosetta.memory"):
            log_memory("test-label")
    assert "test-label" in caplog.text


def test_check_memory_headroom_ok():
    mem = MagicMock()
    mem.available = 8 * 1024**3  # 8 GB free
    with patch("psutil.virtual_memory", return_value=mem):
        check_memory_headroom(min_free_gb=2.0, abort_gb=0.5)  # should not raise


def test_check_memory_headroom_abort():
    mem = MagicMock()
    mem.available = 0.5 * 1024**3  # 0.5 GB — below abort threshold
    with patch("psutil.virtual_memory", return_value=mem):
        with pytest.raises(MemoryError, match="Critical"):
            check_memory_headroom(min_free_gb=2.0, abort_gb=0.75)


def test_check_memory_headroom_warn(caplog):
    mem = MagicMock()
    mem.available = 1.0 * 1024**3  # 1 GB — below warn threshold
    with patch("psutil.virtual_memory", return_value=mem):
        import logging
        with caplog.at_level(logging.WARNING, logger="rosetta.memory"):
            check_memory_headroom(min_free_gb=2.0, abort_gb=0.5)
    assert "Low memory" in caplog.text
