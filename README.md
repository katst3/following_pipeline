## Overview
This framework provides a unified pipeline for training and evaluating models for binary question answering tasks. It supports both TensorFlow and PyTorch backends with LSTM and Transformer architectures respectively, and can work with multiple datasets including combined multi-category datasets.

## Key Features
- **Two Models**: TensorFlow (LSTM) and PyTorch (Transformer)
- **Multi-Dataset Handling**: Supports 2-directional (2dir) and 4-directional (4dir) story datasets
- **Category Support**: Works with individual categories (simple, deeper, less_dense, dense, superdense) and combined datasets
- **Three-Metadata Structure**: Preserves category, num_nouns, and n_actors metadata throughout the pipeline
- **End-to-End Pipeline**: Data processing, model training, optimization, and evaluation
- **Experiment Tracking**: Neptune integration for monitoring performance with detailed category metrics
- **Consistent Metrics Tracking**: Aligned logger implementations for both frameworks



## Dataset Structure

The framework uses pickle files with a three-metadata structure:

1. Category: The density type (simple, deeper, less_dense, dense, superdense)
2. num_nouns: Number of nouns in the story
3. n_actors: Number of actors in the story

Combined datasets (all_2dir, all_4dir) preserve the original category information while providing a unified dataset for training.

### Data Processing
- Load story datasets from pickle files (individual or combined)
- Handle three-metadata structure (category, num_nouns, n_actors)
- Apply name substitutions for balanced representation
- Split data into training, validation, and test sets
- Transform stories into question-answering format


### Model Architectures
- **LSTM**: LSTM-based with separate encoders for stories and questions
- **Transformer**: Transformer-based with positional encoding

### Training
- Configurable hyperparameters (batch size, learning rate, etc.)
- Framework-specific optimizers (Adam for both TF and PT)
- Performance tracking across categories and complexity levels
- Detailed metrics breakdown by category and noun count
- Consistent logging methodology across both frameworks

### Neptune Logging
- Both frameworks use aligned logger implementations:
  - `TransformerNeptuneLogger`: For PyTorch models
  - `LSTMNeptuneLogger`: For TensorFlow models
- Consistent metrics tracking with identical namespace hierarchies
- Detailed tracking by category and noun count combinations
- Visual indicators for best epochs and model improvements
- Robust handling of different metadata formats

### Hyperparameter Optimization
- Ax integration for efficient parameter tuning
- Best models saved automatically with metadata

### Evaluation
- Common metrics across frameworks (accuracy, loss)
- Analysis by story category and complexity
- Support for multiple evaluation sets (valid, validb, test)

