import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a root logger with INFO level, console handler,
    and a consistent formatting style.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
