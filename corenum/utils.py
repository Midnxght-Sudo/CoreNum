import logging

def setup_logger(name: str = "CoreNum") -> logging.Logger:
    """Sets up a logging interface for the library."""
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if the logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Output to the terminal
        handler = logging.StreamHandler()
        
        # Formatting: e.g., "[WARNING] CoreNum: Matrix copy triggered."
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger

# Initialize the global logger for the library to use
logger = setup_logger()