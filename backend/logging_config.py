import json
import logging
import sys
from datetime import datetime, timezone
from logging import LogRecord

from backend.config import settings


class StructuredFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(level: str = "INFO") -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    return root_logger


logger = logging.getLogger(__name__)
