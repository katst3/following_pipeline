# Training Directory

This directory contains components for training and evaluating binary QA models across multiple frameworks. It provides a unified training pipeline that works with both LSTM and Transformer models.

## Contents

- **metrics.py**: Framework-agnostic metrics calculation and analysis
- **model_trainer.py**: Abstract base class for model trainers
- **transformer_neptune_logger.py**: Neptune experiment tracking for PyTorch Transformer models
- **transformer_trainer.py**: PyTorch-specific training implementation for Transformer models
- **lstm_neptune_logger.py**: Neptune logger for TensorFlow/Keras LSTM models
- **lstm_trainer.py**: TensorFlow-specific training implementation for LSTM models

## Architecture Overview

The training system follows a flexible architecture that enables consistent training and evaluation across frameworks:

1. **Abstract Base Class** (`ModelTrainer`):
   - Defines common interface for training
   - Handles experiment tracking initialization
   - Ensures consistent model saving and loading

2. **Framework-Specific Trainers**:
   - `LSTMTrainer`: Implements training for LSTM models
   - `TransformerTrainer`: Implements training for Transformer models

3. **Experiment Tracking**:
   - Neptune integration for logging metrics, hyperparameters, and model artifacts
   - Framework-specific implementations with aligned architecture:
     - `TransformerNeptuneLogger`: Custom logger for PyTorch models
     - `LSTMNeptuneLogger`: Keras callback-based logger for TensorFlow models
   - Both loggers track identical metrics with consistent namespaces in Neptune
   - Both provide detailed tracking by category and noun count
   - Both implement consistent console output with formatted separators

4. **Metrics Processing**:
   - Common interface for evaluating models
   - Framework-specific calculations
   - Support for detailed error analysis

## Training Flow

The training process follows these steps:

1. **Initialization**:
   - Create appropriate trainer (LSTM or Transformer)
   - Configure experiment tracking
   - Initialize appropriate Neptune logger

2. **Training Loop**:
   - LSTM: Uses Keras' fit with callbacks, including the LSTMNeptuneLogger
   - Transformer: Implements manual epoch loop with optimization steps and TransformerNeptuneLogger

3. **Metrics Tracking**:
   - Log metrics after each epoch
   - Track overall metrics (loss, accuracy)
   - Track detailed metrics by category and story complexity
   - Monitor best and top-performing epochs with consistent formatting

4. **Model Checkpointing**:
   - Save best models based on validation accuracy
   - Save periodic checkpoints and top-performing models
   - Log model artifacts to Neptune in consistent namespaces

5. **Evaluation and Analysis**:
   - Calculate detailed metrics using MetricsProcessor
   - Generate confusion matrices
   - Perform error analysis on misclassified examples

## Key Components

### ModelTrainer

Abstract base class that defines the training interface:

```python
def train(self, model, train_data, val_data, metadata_train, metadata_val, hyperparams, epochs=100):
    """Train the model and return validation metrics."""
    pass
```

### Framework-Specific Optimization Approaches

The two trainers reflect fundamental differences in how TensorFlow and PyTorch handle optimization:

1. **TensorFlow/Keras (LSTM Model)**:
   - Loss function and optimizer are defined during model creation
   - The model is pre-compiled with `binary_crossentropy` loss and `Adam` optimizer
   - Training uses the Keras `fit()` method which internally handles the optimization loop

2. **PyTorch (Transformer Model)**:
   - Loss function (`BCEWithLogitsLoss`) and optimizer (`Adam`) are explicitly created in the trainer
   - Manual implementation of forward pass, loss calculation, backpropagation, and optimization steps
   - More explicit control over the training process

### LSTMTrainer

Implements training for TensorFlow LSTM models:

- Uses Keras' fit method with callbacks
- Leverages LSTMNeptuneLogger for metric logging
- Tracks best and top-performing epochs
- Implements detailed progress tracking with TQDM
- Provides consistent console output with formatted separators

### TransformerTrainer

Implements training for PyTorch Transformer models:

- Custom training loop with gradient updates
- Manual validation loop
- Uses TransformerNeptuneLogger for metric logging
- Implements detailed progress tracking with TQDM
- Provides consistent console output with formatted separators

### Neptune Integration

Both frameworks implement Neptune logging with consistent metrics structure:

- Basic metrics in `metrics/` namespace (loss, accuracy)
- Performance by story category (density) in detailed hierarchical namespaces
- Performance by complexity (noun count) in detailed hierarchical namespaces
- Best model tracking in `best/` namespace
- Top epochs tracking in `top_epochs/` namespace

## Evaluation System

The evaluation system provides comprehensive assessment of model performance with detailed breakdowns by story types and complexity levels.

### Evaluation Interface

Both trainers implement a consistent evaluation interface:

```python
def evaluate(self, model, eval_data, metadata_eval, eval_set_name="validation"):
    """Evaluate the model with detailed metrics logging."""
    pass
```

This method returns a dictionary containing:
- Overall metrics (loss, accuracy)
- Predictions and true values for further analysis
- Detailed metrics by category and noun count
- Sample counts for each category/noun count combination

### Detailed Metrics Breakdown

Evaluation results are broken down along several dimensions:

1. **By Story Category**:
   - Performance across different story types (simple, dense, superdense, etc.)
   - Helps identify which narrative structures are most challenging

2. **By Complexity (Noun Count)**:
   - Performance across stories with different numbers of actors
   - Reveals how model performance scales with story complexity

3. **By Category/Complexity Combinations**:
   - Fine-grained analysis of specific scenario types
   - Identifies specific weak points in model understanding

### Framework-Specific Evaluation Implementation

1. **LSTM Evaluation**:
   - Uses Keras' `evaluate` and `predict` methods
   - Batched processing for memory efficiency
   - Metadata-aware filtering to analyze specific subsets

2. **Transformer Evaluation**:
   - Implements custom evaluation loop with torch.no_grad()
   - Batch processing with DataLoader
   - Manual accuracy calculation with sigmoid threshold

### Evaluation Output

A complete evaluation produces the following:

1. **Console Output**:
   - Formatted summary with section separators
   - Category-by-category breakdown of performance
   - Visual indicators for significant findings

2. **Detailed Metrics**:
   - Hierarchical metrics logged to Neptune
   - Performance comparisons across model versions
   - Drill-down capabilities for detailed analysis


### Sample Evaluation Output

```
================================================
EVALUATION RESULTS BY CATEGORY:
  - simple/noun_8: Accuracy=0.9245 (from 318 samples)
  - dense/noun_15: Accuracy=0.8732 (from 142 samples)
  - superdense/noun_30: Accuracy=0.7654 (from 81 samples)
================================================

Overall accuracy: 0.8762
Loss: 0.3124
```

## Implementation Notes

1. **Experiment Tracking**:
   - Every training run creates a unique experiment in Neptune
   - Models are saved with dataset name, run ID, and epoch information
   - Best models are automatically logged as artifacts
   - Metrics are organized in consistent namespace hierarchies

2. **Performance Metrics**:
   - Both frameworks track identical metrics for fair comparison
   - Detailed breakdowns help identify strengths and weaknesses by category and complexity level
   - Visual indicators in console output for new best epochs and top-3 models

3. **Model Saving**:
   - LSTM: Models saved as .weight.h5 files
   - Transformer: Models saved as .pt files
   - Both store in the same models directory with consistent naming
   - Both implement fallback mechanisms for handling errors during saving

4. **Error Handling**:
   - Comprehensive error handling in both implementations
   - Detailed traceback printing for debugging
   - Fallback mechanisms for model saving and metadata processing

5. **Testing Strategies**:
   - Stratified test sets to ensure coverage of all story types
   - Progressive difficulty testing to measure performance boundaries
