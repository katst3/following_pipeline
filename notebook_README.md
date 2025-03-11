# Direction-Following Dataset and QA Framework

This repository contains tools for generating, processing, and evaluating models on binary QA reasoning tasks. The framework consists of two main components:

1. **Dataset Generator** (`data-tutorial.ipynb`): Creates custom datasets with stories about actors moving in different directions. Based on the **qDisCoCirc-inTask** experiment by **Quantinuum**.
2. **QA Framework** (`QA_tutorial_full.ipynb`): Trains and evaluates models on binary question-answering tasks derived from these directional stories.

## Overview

The project focuses on two- and four-directional reasoning capabilities, where stories describe actors moving in various directions (either 2 or 4 directions). The task is to determine whether two actors end up facing the same direction. 

## Workflow

### Step 0: Configure Neptune for Experiment Tracking

Before running any notebooks, update the `config.yaml` file with your Neptune credentials:

```yaml
neptune:
  project_name: "yourprojectname"
  api_token: "yourAPItoken"
```

### Step 1: Generate Direction-Following Datasets

First, use `data-tutorial.ipynb` to create datasets:

1. **Dataset Generation**: Generate stories with actors moving in 2 or 4 directions.
   - Set density parameters that control complexity
   - Configure actor counts for different splits (train / val (valid A) / valb (Valid Comp) /test)
   - Generate positive and negative examples (same/different final directions)

2. **Dataset Inspection**: Verify dataset properties:
   - Story counts in each split
   - Actor count distributions
   - Density categories

3. **Dataset Merging**: Combine datasets by direction type:
   - Create `all_2dir.pkl` for 2-directional datasets
   - Create `all_4dir.pkl` for 4-directional datasets

4. **Validation**: Check that merged datasets maintain correct properties.

### Step 2: Train and Evaluate Models

Next, use `QA_tutorial_full.ipynb` to process the datasets and train models:

1. **Environment Setup**: Import libraries and set random seeds for reproducibility.

2. **Configuration**: Load hyperparameters from `config.yaml`:
   - Select dataset type (`all_2dir` or `all_4dir`)
   - Choose model framework (`lstm` or `transformer`)
   - Decide whether to run hyperparameter optimization

3. **Data Preparation**: Process stories into a format suitable for training:
   - Split into train / val (valid A) / valb (Valid Comp) /test sets
   - Tokenize text and create vocabulary
   - Convert to model-specific data formats

4. **Model Training**:
   - Create model with appropriate architecture
   - Train using either:
     - Direct training with predefined hyperparameters
     - Hyperparameter optimization to find optimal configuration

5. **Evaluation**: Assess model performance:
   - Evaluate on train, validation, and test sets
   - Generate visualizations of performance by actor count and density
   - Analyze where the model succeeds and fails

## Usage Instructions

### Prerequisites

- Python 3.8+
- Required packages: requitements.txt
- Neptune account for experiment tracking

### Setup

Install dependencies:
```
pip install -r requirements.txt
```

### Running the Data Generator

1. Open `data-tutorial.ipynb` in Jupyter Notebook/Lab
2. Configure output directories and parameters
3. Run cells sequentially to:
   - Generate datasets
   - Inspect dataset properties
   - Merge datasets by direction type

The notebook will create:
- Individual dataset files (e.g., `simple_4dir.pkl`, `deeper_2dir.pkl`)
- Merged dataset files (`all_2dir.pkl`, `all_4dir.pkl`)
- Index files for the splits

### Training and Evaluating Models

1. Open `QA_tutorial_full.ipynb` in Jupyter Notebook/Lab
2. Configure:
   - Selected dataset (`all_2dir` or `all_4dir`)
   - Model framework (`lstm` or `transformer`)
   - Whether to run optimization or direct training
3. Run cells sequentially to:
   - Load and process datasets
   - Train models
   - Evaluate and visualize results

## Dataset Structure

Each dataset contains stories structured as:
- **Sentences**: Sequence of statements about actors moving/turning
- **Query**: Question about whether two actors face the same direction
- **Answer**: Binary (yes/no) for whether actors face the same direction
- **Metadata**: Actor count, density etc.

## Models

The framework supports two types of models:

### LSTM Model
- TensorFlow-based implementation
- Processes story and question with separate encoders
- Combines representations for binary classification

### Transformer Model
- PyTorch-based implementation
- Uses transformer encoder architecture
- Jointly processes story and question with attention mechanisms