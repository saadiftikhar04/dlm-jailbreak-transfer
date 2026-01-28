"""
Utility functions for jailbreak experiments.
"""

import logging
from pathlib import Path


def setup_logging(log_file, level='INFO'):
    """Setup logging configuration."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, level))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_checkpoint(data, filepath):
    """Save checkpoint data."""
    import pickle
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def process_output(output, **kwargs):
    """Process model output - placeholder for compatibility."""
    return output
