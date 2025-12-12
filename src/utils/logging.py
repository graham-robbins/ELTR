"""
Logging configuration for IRP platform.

Provides consistent logging setup across all modules with
file and console handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.config import get_config, IRPConfig


def setup_logging(config: IRPConfig | None = None) -> logging.Logger:
    """
    Configure logging for IRP platform.

    Parameters
    ----------
    config : IRPConfig | None
        Configuration object. Uses global config if None.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    if config is None:
        config = get_config()

    log_config = config.logging

    log_file = Path(log_config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_config.level.upper(), logging.INFO)

    root_logger = logging.getLogger("irp")
    root_logger.setLevel(level)

    root_logger.handlers.clear()

    formatter = logging.Formatter(log_config.format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Parameters
    ----------
    name : str
        Module name for logger.

    Returns
    -------
    logging.Logger
        Module-specific logger.
    """
    return logging.getLogger(f"irp.{name}")
