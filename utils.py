# utils.py
"""
Utility functions for the simulation framework.

This module provides helper functions, such as logging setup, that are
used across different parts of the application but do not belong to a
specific domain like physics or rendering.
"""
import logging
import logging.handlers
import json
import os
from typing import Dict, Any

# --- Data Contracts ---
#
# setup_logging(config: Dict[str, Any]) -> None:
#   - Inputs:
#     - config: A dictionary containing a "logging" key with "level",
#       "format", and "log_file" sub-keys.
#   - Outputs: None
#   - Side Effects: Configures the root Python logger. Creates a log
#     directory if it doesn't exist. Sets up a console handler and a
#     rotating file handler.
#   - Invariants: After this function runs, the logging system is
#     initialized and ready for use throughout the application.

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configures the logging system from a configuration dictionary.

    Sets up logging to both the console and a rotating file.
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_file_path = log_config.get('log_file', 'logs/simulation.log')

    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    # Rotates when the log reaches 1MB, keeps 5 backup logs.
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Logging system initialized.")
    logging.debug(f"Log level set to {log_level}.")
    logging.debug(f"Log file path: {log_file_path}")

def load_config(path: str) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    logging.info(f"Loading configuration from {path}...")
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {path}.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {path}.")
        raise