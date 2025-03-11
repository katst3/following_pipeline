# Utils Directory

This directory contains utility modules that support the binary question answering framework across various components. These utilities handle configuration management, experiment tracking, and logging.

## Contents

- **config.py**: Configuration loading, saving, and management
- **experiment_tracker.py**: Neptune integration for tracking experiments
- **logger.py**: Logging setup and formatted output utilities

## Core Utilities

### Configuration Management (`config.py`)

The configuration module provides tools for working with configuration files in YAML and JSON formats:

1. **Loading Configurations**:
   - `load_config(config_path)`: Loads configuration from YAML or JSON files
   - Supports different file formats with consistent interface

2. **Saving Configurations**:
   - `save_config(config, config_path)`: Saves configuration to YAML or JSON files
   - Creates necessary directories automatically

3. **Updating Configurations**:
   - `update_config(config, updates)`: Recursively updates configuration dictionaries
   - Preserves existing values for keys not specified in updates

This module enables consistent configuration handling throughout the framework, allowing for flexible parameter management across different experiments.

### Experiment Tracking (`experiment_tracker.py`)

The experiment tracking module provides a unified interface to Neptune for tracking experiments:

1. **Initialization**:
   - `ExperimentTracker(project_name, api_token)`: Creates a tracker for a specific Neptune project

2. **Run Management**:
   - `start_run(run_name, parameters, tags)`: Starts a new experiment run with specified parameters
   - `end_run()`: Properly terminates an experiment run

3. **Metrics Logging**:
   - `log_metric(metric_name, value, step)`: Logs a single metric
   - `log_metrics(metrics_dict, prefix, step)`: Logs multiple metrics from a dictionary

4. **Artifact Management**:
   - `log_model(model_path, model_name)`: Uploads model files to Neptune

This tracker provides a consistent interface for both lstm and transformer experiments, enabling accurate comparisons between different model architectures and configurations.

### Logging (`logger.py`)

The logging module provides utilities for consistent logging across the framework:

1. **Logger Setup**:
   - `setup_logger(name, log_file, level)`: Creates a logger with both file and console output

2. **Formatted Logging**:
   - `log_hyperparameters(logger, hparams)`: Logs hyperparameters with consistent formatting
   - `log_metrics_summary(logger, metrics)`: Outputs metrics with section-based formatting
   - `log_experiment_start(logger, experiment_name, framework, dataset_info)`: Records the start of an experiment
   - `log_experiment_end(logger, experiment_name, results)`: Records the end of an experiment with results

These logging utilities ensure consistent and readable logs that capture important experiment details, making it easier to track results across multiple runs.

## Integration with the Framework

These utilities form the backbone of the framework's infrastructure:

1. **Configuration Management**:
   - Used in the main pipeline to load and manage experiment parameters
   - Enables consistent hyperparameter management

2. **Experiment Tracking**:
   - Used by `TransformerNeptuneLogger` and `LSTMNeptuneLogger` for metrics logging
   - Provides a unified interface for both frameworks
   - Ensures consistent namespace hierarchies in Neptune across frameworks

3. **Logging**:
   - Used throughout the framework to record experiment progress and results
   - Enables clear and consistent reporting with formatted section separators
   - Provides visual indicators for important events (new best epoch, top-3 models)

The utilities directory provides common functionality that is framework-agnostic, enabling consistent processing across the entire QA pipeline.