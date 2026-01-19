"""Utility functions for logging, file operations, and experiment management.

This module provides helper functions for saving results, managing directories,
and handling experiment logging and configuration.
"""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

# Optional dependency
try:
    import wandb
except ImportError:
    wandb = None


def ensure_directory(path: str) -> None:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, file_path: str, indent: int = 2) -> None:
    """Save object to JSON file.
    
    Args:
        obj: Object to serialize
        file_path: Path to save file
        indent: JSON indentation level
    """
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """Load object from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(header: List[str], rows: List[List], file_path: str) -> None:
    """Save data to CSV file.
    
    Args:
        header: Column headers
        rows: Data rows
        file_path: Path to save file
    """
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_csv(file_path: str, has_header: bool = True) -> tuple:
    """Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        has_header: Whether first row contains headers
        
    Returns:
        Tuple of (headers, rows) if has_header=True, else (None, rows)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        if has_header:
            headers = next(reader)
            rows = list(reader)
            return headers, rows
        else:
            rows = list(reader)
            return None, rows


def initialize_wandb_logging(
    project_name: str,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None
) -> Optional[object]:
    """Initialize Weights & Biases logging.
    
    Args:
        project_name: W&B project name
        entity: W&B entity/user or team
        run_name: Custom run name
        config: Configuration dictionary to log
        tags: List of tags for the run
        
    Returns:
        W&B run object if successful, None if W&B not available
        
    Raises:
        RuntimeError: If W&B is requested but not installed
    """
    if wandb is None:
        raise RuntimeError("wandb not installed. pip install wandb to enable logging")
    
    return wandb.init(
        project=project_name,
        entity=entity,
        name=run_name,
        config=config or {},
        tags=tags or [],
        resume="allow"
    )


def finalize_wandb_logging(wandb_run, summary_metrics: Optional[Dict] = None) -> None:
    """Finalize and close W&B logging.
    
    Args:
        wandb_run: W&B run object
        summary_metrics: Optional summary metrics to log
    """
    if wandb_run is not None:
        if summary_metrics:
            for key, value in summary_metrics.items():
                wandb.summary[key] = value
        wandb_run.finish()



def format_time_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m{secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h{int(minutes)}m{secs:.1f}s"


def create_results_summary(
    experiment_config: Dict,
    best_mechanism: Dict,
    best_score: float,
    checkpoint_path: str,
    experiment_duration: float
) -> Dict:
    """Create comprehensive experiment results summary.
    
    Args:
        experiment_config: Experiment configuration
        best_mechanism: Best mechanism parameters found
        best_score: Best objective score achieved
        checkpoint_path: Path to model checkpoint
        experiment_duration: Total experiment time in seconds
        
    Returns:
        Comprehensive results dictionary
    """
    return {
        "experiment_info": {
            "completion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": experiment_duration,
            "duration_formatted": format_time_duration(experiment_duration),
        },
        "best_results": {
            "mechanism_parameters": best_mechanism,
            "objective_score": float(best_score),
            "model_checkpoint": str(checkpoint_path),
        },
        "experiment_configuration": experiment_config,
    }