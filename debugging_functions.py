import pickle
import os 
import glob 
import json
import pandas as pd

def inspect_dataset_actor_counts(file_path):
    """Inspect actor counts by split for a dataset file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    split_actor_counts = {'train': {}, 'valid': {}, 'validB': {}, 'test': {}}
    
    for key, group in data.items():
        if isinstance(key, tuple) and len(key) > 0:
            split = key[0]
            if split not in split_actor_counts:
                continue
                
            for story_type in ['pos', 'neg']:
                for story in group[story_type]:
                    if len(story) > 2 and isinstance(story[2], dict):
                        metadata = story[2]
                        if 'num_actors' in metadata:
                            actor_count = metadata['num_actors']
                            if actor_count not in split_actor_counts[split]:
                                split_actor_counts[split][actor_count] = 0
                            split_actor_counts[split][actor_count] += 1
    
    print(f"Dataset file: {os.path.basename(file_path)}")
    for split, actor_counts in split_actor_counts.items():
        if not actor_counts:
            continue
            
        print(f"  {split} split actor counts:")
        for actor_count in sorted(actor_counts.keys()):
            count = actor_counts[actor_count]
            print(f"    - {actor_count} actors: {count} stories")
    print()


def enhanced_merge_datasets_by_direction(data_dir, direction_type):
    """
    Enhanced version of merge_datasets_by_direction with actor count tracking.
    
    Args:
        data_dir (str): Directory containing dataset files
        direction_type (str): Direction type ('2dir' or '4dir')
        
    Returns:
        dict: Configuration for the combined dataset
    """
    print(f"Combining {direction_type} datasets...")
    
    pickle_pattern = os.path.join(data_dir, f"*_{direction_type}.pkl")
    dataset_files = glob.glob(pickle_pattern)

    dataset_files = [f for f in dataset_files if not os.path.basename(f).startswith("all_")]
    
    if not dataset_files:
        print(f"No {direction_type} datasets found in {data_dir}")
        return None
    
    print(f"Found {len(dataset_files)} datasets: {[os.path.basename(f) for f in dataset_files]}")
    
    combined_stories = {}
    
    split_mappings = {
        'validA': 'valid',  # Map any 'validA' to 'valid'
        'validB': 'validb',  # Map any 'validB' to 'validb'
        'valid': 'valid',    
        'train': 'train',    
        'validb': 'validb',  
        'test': 'test'       
    }
    
    split_stories = {
        'train': [],
        'valid': [],
        'validb': [],
        'test': []
    }
    
    actor_counts_before_merge = {'train': {}, 'valid': {}, 'validB': {}, 'test': {}}
    actor_counts_after_merge = {'train': {}, 'valid': {}, 'validb': {}, 'test': {}}
    
    for dataset_file in dataset_files:
        dataset_name = os.path.basename(dataset_file).replace('.pkl', '')
        density_type = dataset_name.split('_')[0]
        print(f"\nProcessing {dataset_name} (density type: '{density_type}')...")
        
        with open(dataset_file, 'rb') as f:
            stories = pickle.load(f)
        
        for key, group in stories.items():
            if isinstance(key, tuple) and len(key) > 0:
                original_split = key[0]
                
                for story_type in ['pos', 'neg']:
                    for story in group[story_type]:
                        if len(story) > 2 and isinstance(story[2], dict):
                            metadata = story[2]
                            if 'num_actors' in metadata:
                                actor_count = metadata['num_actors']
                                if actor_count not in actor_counts_before_merge[original_split]:
                                    actor_counts_before_merge[original_split][actor_count] = 0
                                actor_counts_before_merge[original_split][actor_count] += 1
                
        for key, group in stories.items():
            if isinstance(key, tuple) and len(key) > 0:
                original_split = key[0]
                std_split = split_mappings.get(original_split, original_split.lower())
                new_key = key[1:] if len(key) > 1 else key
            else:
                print(f"  Warning: Key {key} does not contain split information, skipping")
                continue
            
            for story in group["pos"]:
                metadata = {}
                if len(story) > 2 and isinstance(story[2], dict):
                    metadata = dict(story[2])
                
                metadata['category'] = density_type
                
                updated_story = (story[0], story[1], metadata)
                
                split_stories[std_split].append((new_key, "pos", updated_story))
            
            for story in group["neg"]:
                metadata = {}
                if len(story) > 2 and isinstance(story[2], dict):
                    metadata = dict(story[2])
                
                metadata['category'] = density_type
                
                updated_story = (story[0], story[1], metadata)
                
                split_stories[std_split].append((new_key, "neg", updated_story))
    
    for split, stories_list in split_stories.items():
        for story_info in stories_list:
            _, _, story = story_info
            if len(story) > 2 and isinstance(story[2], dict):
                metadata = story[2]
                if 'num_actors' in metadata:
                    actor_count = metadata['num_actors']
                    if actor_count not in actor_counts_after_merge[split]:
                        actor_counts_after_merge[split][actor_count] = 0
                    actor_counts_after_merge[split][actor_count] += 1
    
    print("\nActor count comparison before and after merging:")
    for split in ['train', 'valid', 'validB', 'validb', 'test']:
        if split in actor_counts_before_merge and actor_counts_before_merge[split]:
            print(f"\n{split} split before merging:")
            for actor_count in sorted(actor_counts_before_merge[split].keys()):
                count = actor_counts_before_merge[split][actor_count]
                print(f"  - {actor_count} actors: {count} stories")
    
    for split in ['train', 'valid', 'validb', 'test']:
        if actor_counts_after_merge[split]:
            print(f"\n{split} split after merging:")
            for actor_count in sorted(actor_counts_after_merge[split].keys()):
                count = actor_counts_after_merge[split][actor_count]
                print(f"  - {actor_count} actors: {count} stories")
    
    next_global_index = 0
    split_indices = {split: [] for split in split_stories}
    
    for split, stories_list in split_stories.items():
        for story_info in stories_list:
            key, story_type, story = story_info
            
            if key not in combined_stories:
                combined_stories[key] = {"pos": [], "neg": []}
            
            combined_stories[key][story_type].append(story)
            
            split_indices[split].append(next_global_index)
            next_global_index += 1
    
    combined_name = f"all_{direction_type}"
    combined_stories_path = os.path.join(data_dir, f"{combined_name}.pkl")
    with open(combined_stories_path, 'wb') as f:
        pickle.dump(combined_stories, f)
    
    for split in ['train', 'valid', 'validb', 'test']:
        indices_path = os.path.join(data_dir, f"{split}_indices_{combined_name}.json")
        with open(indices_path, 'w') as f:
            json.dump(split_indices[split], f)
        print(f"Saved {len(split_indices[split])} {split} indices to {indices_path}")
    
    print(f"\nSuccessfully combined datasets into {combined_name}")
    print(f"Combined dataset has {len(split_indices['train'])} train, " + 
          f"{len(split_indices['valid'])} valid, " + 
          f"{len(split_indices['validb'])} validb, and " + 
          f"{len(split_indices['test'])} test examples")
    
    return {
        'stories_file': combined_stories_path,
        'train_indices_file': os.path.join(data_dir, f"train_indices_{combined_name}.json"),
        'valid_indices_file': os.path.join(data_dir, f"valid_indices_{combined_name}.json"),
        'validb_indices_file': os.path.join(data_dir, f"validb_indices_{combined_name}.json"),
        'test_indices_file': os.path.join(data_dir, f"test_indices_{combined_name}.json")
    }

def trace_evaluation_data_flow(data_info, eval_results, stage_name):
    """
    Trace the data flow at key pipeline stages to verify data consistency.
    
    Args:
        data_info: Dictionary containing dataset information
        eval_results: Dictionary containing evaluation results
        stage_name: String identifying the current pipeline stage
    """
    print(f"\n{'='*20} DATA FLOW TRACE: {stage_name} {'='*20}")
    
    datasets = []
    if 'train_df' in data_info: datasets.append('Train')
    if 'val_df' in data_info: datasets.append('Valid A')
    if 'valb_df' in data_info: datasets.append('Valid Comp')
    if 'test_df' in data_info: datasets.append('Test')
    print(f"Available datasets: {datasets}")
    
    for df_key in ['train_df', 'val_df', 'valb_df', 'test_df']:
        if df_key in data_info and isinstance(data_info[df_key], pd.DataFrame):
            df = data_info[df_key]
            print(f"\nDataFrame '{df_key}': {len(df)} rows with columns {list(df.columns)}")
            
            if 'category' in df.columns:
                categories = sorted(df['category'].unique())
                print(f"  Categories ({len(categories)}): {categories}")
                for category in categories:
                    cat_count = len(df[df['category'] == category])
                    print(f"    - {category}: {cat_count} rows ({cat_count/len(df)*100:.1f}%)")
            else:
                print(f"  WARNING: No 'category' column in {df_key}!")
            
            for col in ['num_actors', 'n_actors', 'num_nouns']:
                if col in df.columns:
                    actor_counts = sorted(df[col].unique())
                    print(f"  {col} values ({len(actor_counts)}): {actor_counts}")
                    
                    if 'category' in df.columns:
                        print(f"  Actor counts by category:")
                        for category in categories:
                            cat_df = df[df['category'] == category]
                            cat_actors = sorted(cat_df[col].unique())
                            print(f"    - {category}: {cat_actors}")
                    break
            else:
                print(f"  WARNING: No actor count column in {df_key}!")
    
    print(f"\nEvaluation results structure:")
    if not isinstance(eval_results, dict):
        print(f"  WARNING: eval_results is not a dictionary! Type: {type(eval_results)}")
        return
    
    print(f"  Keys: {list(eval_results.keys())}")
    
    if 'accuracy' in eval_results:
        print(f"  Overall accuracy: {eval_results['accuracy']:.4f}")
    if 'loss' in eval_results:
        print(f"  Overall loss: {eval_results['loss']:.4f}")
    
    for key_prefix in ['train', 'valid', 'validation', 'test', 'validb']:
        acc_key = f"{key_prefix}_accuracy"
        if acc_key in eval_results:
            print(f"  {acc_key}: {eval_results[acc_key]:.4f}")
    
    for key in ['train_accuracies', 'validation_accuracies', 'validb_accuracies', 'test_accuracies']:
        if key in eval_results and eval_results[key]:
            print(f"\n  {key}:")
            for category, actor_data in eval_results[key].items():
                print(f"    Category '{category}':")
                print(f"      Actor counts: {sorted(actor_data.keys())}")
                
                sample_counts = sorted(actor_data.keys())[:3]
                for count in sample_counts:
                    print(f"      Actor count {count}: accuracy = {actor_data[count]:.4f}")
    
    if 'detailed_metrics' in eval_results:
        print(f"\n  detailed_metrics:")
        for category, counts in eval_results['detailed_metrics'].items():
            print(f"    Category '{category}':")
            count_keys = list(counts.keys())
            print(f"      Keys format: {count_keys[:3]} ...")
            
            for key in count_keys[:3]:
                print(f"      {key}: {counts[key]}")
    
    if 'sample_sizes' in eval_results:
        print(f"\n  sample_sizes:")
        for dataset, categories in eval_results['sample_sizes'].items():
            print(f"    {dataset}:")
            for category, sizes in categories.items():
                print(f"      {category}: {sizes}")
    
    print(f"{'='*60}\n")