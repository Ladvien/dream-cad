import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

class TUILogger:
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path.home() / ".dreamcad" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"dreamcad_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        self.logger = logging.getLogger("dreamcad")
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        self.logger.info("DreamCAD TUI Logger initialized")
        
    def debug(self, message: str) -> None:
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        self.logger.warning(message)
        
    def error(self, message: str, exc_info: bool = True) -> None:
        self.logger.error(message, exc_info=exc_info)
        
    def critical(self, message: str) -> None:
        self.logger.critical(message)
        
    def get_log_file(self) -> Path:
        return self.log_file