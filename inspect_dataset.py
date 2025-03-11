"""
script to inspect a pickled dataset file and verify category information.

usage:
python inspect_pickle.py datasets/all_2dir.pkl
"""

# this is a utility script to examine the structure of story dataset files
# it helps ensure the category metadata is properly preserved and distributed
# useful for debugging category-related issues before training models

import sys
import pickle
import random
from collections import Counter

def inspect_pickle(pickle_path):
    """inspect a pickled dataset file and verify category information."""
    print(f"Inspecting pickle file: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Dataset contains {len(data)} keys")
    
    sample_keys = random.sample(list(data.keys()), min(5, len(data)))
    
    categories = Counter()
    stories_with_metadata = 0
    stories_without_metadata = 0
    total_stories = 0
    
    # inspect the structure and check category information
    for key in sample_keys:
        print(f"\nInspecting key: {key}")
        pos_stories = data[key]['pos']
        neg_stories = data[key]['neg']
        
        print(f"  - Positive stories: {len(pos_stories)}")
        print(f"  - Negative stories: {len(neg_stories)}")
        
        if pos_stories:
            sample_pos = pos_stories[0]
            print("\nSample positive story:")
            if len(sample_pos) > 2 and isinstance(sample_pos[2], dict):
                print(f"  Has metadata: {sample_pos[2]}")
                if 'category' in sample_pos[2]:
                    print(f"  Category: {sample_pos[2]['category']}")
            else:
                print("  No metadata found")
        
        if neg_stories:
            sample_neg = neg_stories[0]
            print("\nSample negative story:")
            if len(sample_neg) > 2 and isinstance(sample_neg[2], dict):
                print(f"  Has metadata: {sample_neg[2]}")
                if 'category' in sample_neg[2]:
                    print(f"  Category: {sample_neg[2]['category']}")
            else:
                print("  No metadata found")
    
    # count all categories in the dataset
    print("\nCounting all categories in the dataset...")
    for key, value in data.items():
        for story_list in [value['pos'], value['neg']]:
            for story in story_list:
                total_stories += 1
                if len(story) > 2 and isinstance(story[2], dict) and 'category' in story[2]:
                    stories_with_metadata += 1
                    categories[story[2]['category']] += 1
                else:
                    stories_without_metadata += 1
    
    print(f"\nTotal stories: {total_stories}")
    print(f"Stories with metadata: {stories_with_metadata} ({stories_with_metadata/total_stories*100:.1f}%)")
    print(f"Stories without metadata: {stories_without_metadata} ({stories_without_metadata/total_stories*100:.1f}%)")
    
    print("\nCategory distribution:")
    for category, count in categories.most_common():
        print(f"  - {category}: {count} stories ({count/total_stories*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pickle.py <pickle_file>")
        sys.exit(1)
    
    inspect_pickle(sys.argv[1])