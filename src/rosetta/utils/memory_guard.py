import logging
import os

import psutil

log = logging.getLogger("rosetta.memory")


def log_memory(label: str) -> None:
    """Log the current process RSS to the rosetta.memory logger."""
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    log.info("[%s] RSS = %.2f GB", label, rss_gb)


def check_memory_headroom(min_free_gb: float = 2.0) -> None:
    """Warn if available system RAM is below min_free_gb."""
    avail_gb = psutil.virtual_memory().available / 1024**3
    if avail_gb < min_free_gb:
        log.warning(
            "Low memory: only %.1f GB available (threshold %.1f GB). "
            "Consider --concurrency 1 or --max-instructions to reduce load.",
            avail_gb,
            min_free_gb,
        )
