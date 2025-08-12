import logging
import os


def setup_logger(
    name="pipeline", log_file="pipeline.log", level=logging.DEBUG, include_location=True
):
    """Set up and return a logger with file and console handlers."""
    # Anchor logs_dir to the project root (where logger.py is)
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, log_file)

    logger = logging.getLogger(name)
    if logger.handlers:  # Avoid duplicate handlers
        return logger

    logger.setLevel(level)

    # Create formatters
    if include_location:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
            datefmt="%m-%d-%Y %I:%M%p",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%m-%d-%Y %I:%M%p"
        )

    # File handler with append mode and flushing
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    file_handler = FlushingFileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.debug(f"Logger initialized, writing to: {log_path}")
    return logger
