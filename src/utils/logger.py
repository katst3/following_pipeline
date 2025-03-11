# consistent logging locally

import logging
import os
import json
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """Set up logger with file and console handlers."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_hyperparameters(logger, hparams):
    """Log hyperparameters."""
    logger.info("=== Hyperparameters ===")
    for param_name, param_value in hparams.items():
        logger.info(f"  {param_name}: {param_value}")
    logger.info("======================")

def log_metrics_summary(logger, metrics):
    """Log metrics summary."""
    logger.info("=== Metrics Summary ===")
    
    # handle nested metrics dictionaries
    if isinstance(metrics, dict) and any(isinstance(v, dict) for v in metrics.values()):
        for section, section_metrics in metrics.items():
            logger.info(f"  {section}:")
            if isinstance(section_metrics, dict):
                for metric_name, metric_value in section_metrics.items():
                    logger.info(f"    {metric_name}: {metric_value}")
            else:
                logger.info(f"    {section_metrics}")
    else:
        # simple dictionary
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value}")
    
    logger.info("=======================")

def log_experiment_start(logger, experiment_name, framework, dataset_info):
    """Log experiment start information."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting experiment: {experiment_name} at {timestamp}")
    logger.info(f"Framework: {framework}")
    logger.info("Dataset info:")
    for key, value in dataset_info.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")

def log_experiment_end(logger, experiment_name, results):
    """Log experiment end information."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Completed experiment: {experiment_name} at {timestamp}")
    
    if 'best_val_accuracy' in results:
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    
    if 'history' in results and isinstance(results['history'], dict):
        logger.info("Final metrics:")
        for metric_name, metric_value in results['history'].items():
            # handle history over epochs
            if isinstance(metric_value, list):
                logger.info(f"  {metric_name} (final): {metric_value[-1]:.4f}")
            else:
                logger.info(f"  {metric_name}: {metric_value:.4f}")