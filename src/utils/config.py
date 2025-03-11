# handles configuration loading and saving between YAML and JSON formats

import json
import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    elif ext.lower() == '.json':
        with open(config_path, 'r') as file:
            config = json.load(file)
    else:
        raise ValueError(f"Unsupported config file extension: {ext}")
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    elif ext.lower() == '.json':
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
    else:
        raise ValueError(f"Unsupported config file extension: {ext}")

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values."""
    result = config.copy()
    
    def _update_dict_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict_recursive(d[k], v)
            else:
                d[k] = v
        return d
    
    return _update_dict_recursive(result, updates)