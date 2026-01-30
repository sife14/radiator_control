"""
Radiator Control - Hilfsfunktionen
==================================
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Lädt Konfiguration aus YAML-Datei."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path) as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
):
    """Konfiguriert Logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Begrenzt Wert auf Bereich."""
    return max(min_val, min(max_val, value))
