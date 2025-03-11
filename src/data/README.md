# Data Directory

This directory contains modules for loading, processing, and preparing story datasets for the binary QA framework. It handles the transformation from raw story data to vectorized formats ready for both LSTM and Transformer models.

## Contents

- **abstract_preprocessor.py**: Defines the interface for data preprocessing implementations
- **data_utils.py**: Utility functions for dataset analysis and visualization
- **transformer_preprocessor.py**: PyTorch-specific preprocessing for Transformer models
- **story_processor.py**: Core functionality for story dataset processing
- **lstm_preprocessor.py**: TensorFlow-specific preprocessing for LSTM models

## Pipeline Flow

### 1. Story Loading and Processing (`story_processor.py`)

The `StoryProcessor` class handles:

- Loading story datasets from pickle files
- Filtering stories by number of nouns
- Creating structured DataFrame representations
- Splitting data into train/val/valb(valComp)/test sets using predefined indices
- Name substitution for balanced representation
- Transforming stories into question-answering format

The name substitution process is particularly important:
1. Original names are replaced with placeholders to track them
2. Placeholders are replaced with random names, ensuring balanced usage
3. This prevents models from associating specific outcomes with specific names

### 2. Framework-Specific Preprocessing

#### For TensorFlow Models (`lstm_preprocessor.py`)

The `LSTMDataPreprocessor` handles:
- Building vocabulary from stories and questions
- Using Keras Tokenizer to convert words to indices
- Creating padded sequences for stories and questions separately
- Converting "yes"/"no" answers to binary values

Output format: `X_padded`, `Xq_padded`, and binary `Y` arrays

#### For PyTorch Models (`transformer_preprocessor.py`)

The `TransformerDataPreprocessor` handles:
- Building custom vocabulary with special tokens (`<CLS>`, `<SEP>`, `<PAD>`, `<UNK>`)
- Combining story and question with separator tokens
- Creating attention masks for the transformer model
- Implementing the PyTorch Dataset interface for efficient batch loading

Output format: PyTorch tensors for input_ids, attention_masks, and labels

### 3. Analysis and Validation (`data_utils.py`)

Utility functions for:
- Analyzing dataset characteristics (word counts, noun counts)
- Verifying balanced representation of character names
- Checking data distribution across categories and complexity levels
- Visualizing character presence across stories