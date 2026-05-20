import logging
import os

import psutil

log = logging.getLogger("rosetta.memory")


def log_memory(label: str) -> None:
    """Log the current process RSS to the rosetta.memory logger."""
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    log.info("[%s] RSS = %.2f GB", label, rss_gb)


def check_memory_headroom(min_free_gb: float = 2.0, abort_gb: float = 0.75) -> None:
    """Warn if available system RAM is below min_free_gb; abort below abort_gb."""
    avail_gb = psutil.virtual_memory().available / 1024**3
    if avail_gb < abort_gb:
        raise MemoryError(
            f"Critical: {avail_gb:.2f} GB free — aborting to prevent system freeze. "
            f"Re-run with --resume to continue from the partial save."
        )
    if avail_gb < min_free_gb:
        log.warning(
            "Low memory: only %.1f GB available (threshold %.1f GB). "
            "Reduce load with: --concurrency 1 --chunk-size 1 --max-instructions 50",
            avail_gb,
            min_free_gb,
        )
