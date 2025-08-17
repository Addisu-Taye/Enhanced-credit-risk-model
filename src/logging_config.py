import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger("credit_risk")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = RotatingFileHandler(
        'logs/credit_risk.log',
        maxBytes=1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.ERROR)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()