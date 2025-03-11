# Direction Following Dataset Generator

This repository contains a Python script that generates balanced datasets for testing two- and four- directional reasoning capabilities. The dataset consists of stories about actors moving in different directions, with tasks to determine whether two specified actors end up facing the same direction.
Based on the **qDisCoCirc-inTask** experiment by **Quantinuum**.

## Overview

The script generates stories with varying complexity levels where actors walk in different directions and perform actions such as:
- Walking in a specific direction (north, south, east, west)
- Following another actor
- Going in the opposite direction of another actor
- Turning left, right, or around

Each story is paired with a question about whether two randomly selected actors are facing the same direction by the end of the story.

## Dataset Variations

The script generates datasets with different density levels and direction types:

1. **Direction Types**:
   - `2dir`: Only north and south directions
   - `4dir`: All four directions (north, south, east, west)

2. **Density Levels** (categories):
   - `simple`: ~26% density 
   - `deeper`: ~48% density 
   - `less_dense`: ~50% density 
   - `dense`: ~58% density 
   - `superdense`: ~68% density 

Density refers to the proportion of interpersonal actions ('follows', 'opposite direction of') between actors, compared to the total number of actions.

## Dataset Structure

Each dataset is split into four parts with specific actor ranges:
- **Train**: Stories with 2-8 actors
- **Valid**: Stories with 2-8 actors (in-distribution validation)
- **ValidB**: Stories with 9-20 actors (out-of-distribution validation)
- **Test**: Stories with 21-30 actors (extreme out-of-distribution testing)


## Balance Modes

The script supports three different ways to balance the dataset:

1. **Actor-only Balance** (`actor_only`): Ensures an equal distribution of stories across different actor counts.
2. **Density-only Balance** (`density_only`): Ensures an equal distribution of stories across different density categories.
3. **Full Balance** (`both`): Ensures a balanced distribution across both actor counts and density categories.

For each balance category, the script also aims to maintain an equal number of positive examples (actors facing the same direction) and negative examples (actors facing different directions).

## Story Format

Each story in the dataset is represented as a tuple with three elements:
1. A list of sentences describing actor movements (e.g., "Bob walks north. Alice turns right.")
2. A tuple `(actor1_name, actor2_name, boolean)` where the boolean indicates whether the two actors are facing the same direction
3. A metadata dictionary containing:
   - `category`: The density level of the story (e.g., "simple", "dense", "superdense")
   - `num_actors`: The number of unique actors in the story
   - `n_sentences`: The number of sentences in the story

## File Format

The script generates several files:
- Main dataset file: `[category]_[n_directions]dir.pkl` (contains all stories)
- Index files for different splits:
  - `train_indices_[category]_[n_directions]dir.json`
  - `valid_indices_[category]_[n_directions]dir.json`
  - `validB_indices_[category]_[n_directions]dir.json`
  - `test_indices_[category]_[n_directions]dir.json`
- Metadata file: `metadata_[category]_[n_directions]dir.json`
- Combined datasets: `all_[n_directions]dir.pkl` (when merging multiple datasets)

## Classes and Functions

**Main Classes:**
- **Actor**: Represents a character that can move in different directions
- **Story**: Generates sequences of events involving multiple actors with target density

**Key Functions:**
- `track_story()`: Determines the final direction of each actor
- `generate_story()`: Creates a single story with specified parameters
- `generate_balanced_dataset()`: Creates stories with balanced distribution of positive/negative examples
- `generate_complete_dataset()`: Main function to create all dataset splits
- `analyze_dataset()`: Prints detailed statistics about the generated dataset
- `generate_experiment_datasets()`: Generates all datasets from Table 2 in the paper
- `merge_datasets_by_direction()`: Combines multiple datasets with the same direction type

## Usage

You can use the dataset generator either through command-line arguments or by importing the functions directly in Python code or notebooks.

### Python/Notebook Usage

For Python or notebook usage:
```python
# Import the generator
from dataset_generator import *

# Generate a single dataset
dataset_path = generate_complete_dataset(
    train_size=200,
    valid_size=50, 
    validB_size=100,
    test_size=100,
    n_directions=4,
    target_densities=[0.26],  # simple density
    balance_mode="actor_only",
    output_dir="./datasets_new",
    output_name="simple_4dir"
)

# Generate all experiment datasets
generate_experiment_datasets(output_dir="./datasets_new")

# Merge datasets by direction
config_4dir = merge_datasets_by_direction("./datasets_new", "4dir")
```

### Command-line Usage

You can customize the dataset generation process through command-line arguments:

#### Commands

The script supports three main commands:
- `single_dataset`: Generate a single dataset with specified parameters
- `experiment_datasets`: Generate all datasets from Table 2 in the paper
- `merge_datasets`: Combine multiple datasets with the same direction type

#### Single Dataset Parameters

- `--train`: Number of stories for train split (default: 200)
- `--valid`: Number of stories for valid split (default: 40)
- `--validB`: Number of stories for validB split (default: 40)
- `--test`: Number of stories for test split (default: 40)
- `--n_directions`: Number of directions (2 or 4, default: 4)
- `--density`: Target density value (e.g., 0.26, 0.48, 0.50, 0.58, 0.68)
- `--balance_mode`: How to balance the dataset (actor_only, density_only, or both)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Directory to save generated datasets (default: "./datasets")
- `--output_name`: Custom name for the output dataset

### Merge Datasets Parameters

- `--dir_type`: Direction type for merging datasets (2dir or 4dir)

### Command-line Examples

Generate a single dataset with default settings:
```bash
python dataset_generator.py --command single_dataset
```

Generate a dataset with custom parameters:
```bash
python dataset_generator.py --command single_dataset --train 100 --valid 20 --validB 20 --test 20 --n_directions 2 --density 0.26 --balance_mode actor_only
```

Generate all experiment datasets:
```bash
python dataset_generator.py --command experiment_datasets
```

Merge datasets by direction type:
```bash
python dataset_generator.py --command merge_datasets --dir_type 4dir
```

## Requirements

- Python 3.6+
- Required libraries:
  - pickle
  - json
  - os
  - random
  - tqdm
  - argparse

## Output Directory

By default, datasets are saved to "./datasets" as specified by the `--output_dir` argument, but you can override this by changing the argument value. The script will create the directory if it doesn't exist.
