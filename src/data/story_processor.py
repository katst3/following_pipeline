# handles story loading, processing, and metadata during data preprocessing
# maps tuple keys to density categories and ensures category metadata is preserved
# replaces names for uniform distribution

import re
import json
import pickle
import pandas as pd
from collections import Counter
import random

def map_tuple_to_density_category(tuple_key):
    if not isinstance(tuple_key, tuple) and hasattr(tuple_key, '__iter__'):
        try:
            if isinstance(tuple_key, str):
                parts = tuple_key.strip('()').split(',')
                tuple_key = (int(parts[0]), int(parts[1]))
            else:
                tuple_key = tuple(tuple_key)
        except Exception as e:
            print(f"Error converting to tuple: {e}")
            return 'unknown'
    
    num_actors, num_sentences = tuple_key
    

    if num_actors <= 8:
        return 'simple'
    elif num_actors <= 20:
        ratio = num_sentences / num_actors
        
        if ratio < 3:
            return 'deeper'
        elif ratio < 4:
            return 'less_dense'
        else:
            return 'dense'
    else:
        return 'superdense'

class StoryProcessor:
    """Unified class for loading and processing story datasets."""
    
    @staticmethod
    def load_dataset(filename):
        with open(filename, 'rb') as file:
            dataset = pickle.load(file)
            print(f"Loaded dataset with {len(dataset)} tuple keys")
            return dataset
    
    @staticmethod
    def filter_and_organize_stories(self, dataset, category=None, max_nouns=None):
        print(f"Processing all tuple categories and mapping to density categories")
        
        organized_stories = []
        processed_tuples = 0
        category_counts = {}
        
        for tuple_key, value in dataset.items():
            if not isinstance(value, dict) or 'pos' not in value or 'neg' not in value:
                print(f"  Warning: Unexpected format for key {tuple_key}")
                continue
                    
            for story in value['pos']:
                if story and len(story) >= 1:
                    # extract category from metadata if available
                    story_category = None
                    if len(story) > 2 and isinstance(story[2], dict) and 'category' in story[2]:
                        story_category = story[2]['category']
                    else:
                        # map the tuple key to a density category if no metadata
                        story_category = self.map_tuple_to_density_category(tuple_key)
                    
                    organized_stories.append({
                        'story': story,
                        'label': 1,
                        'category': story_category,
                        'story_type': 'positive',
                        'original_tuple': tuple_key,
                        'num_actors': tuple_key[0] if isinstance(tuple_key, tuple) and len(tuple_key) > 0 else 0,
                        'num_sentences': tuple_key[1] if isinstance(tuple_key, tuple) and len(tuple_key) > 1 else 0
                    })
                    
                    if story_category not in category_counts:
                        category_counts[story_category] = 0
                    category_counts[story_category] += 1
            
            for story in value['neg']:
                if story and len(story) >= 1:
                    story_category = None
                    if len(story) > 2 and isinstance(story[2], dict) and 'category' in story[2]:
                        story_category = story[2]['category']
                    else:
                        story_category = self.map_tuple_to_density_category(tuple_key)
                    
                    organized_stories.append({
                        'story': story,
                        'label': 0,
                        'category': story_category,
                        'story_type': 'negative',
                        'original_tuple': tuple_key,
                        'num_actors': tuple_key[0] if isinstance(tuple_key, tuple) and len(tuple_key) > 0 else 0,
                        'num_sentences': tuple_key[1] if isinstance(tuple_key, tuple) and len(tuple_key) > 1 else 0
                    })
                    
                    if story_category not in category_counts:
                        category_counts[story_category] = 0
                    category_counts[story_category] += 1
            
            processed_tuples += 1
        
        print(f"  Processed {processed_tuples} tuple keys into {len(organized_stories)} stories")
        
        print("  Density category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"    - {cat}: {count} stories ({count/len(organized_stories)*100:.1f}%)")
        
        return organized_stories

    def create_dataframe(self, merged_dataset):
        # convert datasets to dataframe format with consistent metadata
        if isinstance(merged_dataset, list) and len(merged_dataset) > 0 and isinstance(merged_dataset[0], dict):
            df = pd.DataFrame(merged_dataset)
            print(f"Created DataFrame with {len(df)} story dictionaries")
        else:
            # handle old format (tuple format)
            data = []
            for item in merged_dataset:
                if isinstance(item, tuple) and len(item) >= 4:
                    key, pos_neg, story, category = item
                    if isinstance(key, tuple) and len(key) >= 2:
                        num_actors, num_sentences = key[0], key[1]
                        
                        metadata = {}
                        if len(story) > 2 and isinstance(story[2], dict):
                            metadata = story[2]
                        
                        story_category = metadata.get('category', category)
                        
                        data.append({
                            'story': story,
                            'num_actors': num_actors, 
                            'num_sentences': num_sentences,
                            'story_type': pos_neg,
                            'category': story_category,
                            'original_tuple': key,
                            'label': 1 if pos_neg == 'pos' else 0
                        })
                    else:
                        print(f"Warning: Unexpected key format: {key}")
                
            df = pd.DataFrame(data)
            print(f"Created DataFrame from {len(df)} stories in legacy tuple format")
                
        # ensure required columns exist
        if 'category' not in df.columns:
            print("Warning: No 'category' column found, adding default")
            df['category'] = 'unknown'
                
        if 'story_type' not in df.columns:
            print("Warning: No 'story_type' column found, adding default")
            if 'label' in df.columns:
                df['story_type'] = df['label'].apply(lambda x: 'positive' if x == 1 else 'negative')
            else:
                df['story_type'] = 'unknown'
        
        # print category distribution for debugging
        if 'category' in df.columns:
            categories = df['category'].value_counts()
            print("Category distribution in DataFrame:")
            for category, count in categories.items():
                print(f"  - {category}: {count} stories ({count/len(df)*100:.1f}%)")
        
        return df
    
    @staticmethod
    def read_indices_from_json(filename):
        with open(filename, 'r') as file:
            indices = json.load(file)
            print(f"Read {len(indices)} indices from {filename}")
            return indices
    
    @staticmethod
    def split_dataframe(df, indices):
        # filter out indices that are out of bounds
        valid_indices = [idx for idx in indices if idx < len(df)]
        if len(valid_indices) < len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} indices are out of bounds")
            
        split_df = df.iloc[valid_indices]
        
        # print category distribution for debugging
        if 'category' in split_df.columns:
            categories = split_df['category'].value_counts()
            print("Category distribution in split DataFrame:")
            for category, count in categories.items():
                print(f"  - {category}: {count} stories ({count/len(split_df)*100:.1f}%)")
                
        return split_df
    
    @staticmethod
    def replace_names_with_placeholders(story_text, story_tuple, names, story_index):
        # replace all names with unique placeholders to allow controlled name swapping
        placeholders = {}
        new_story_tuple = list(story_tuple)

        for name in names:
            placeholder = f"PLACEHOLDER_{name}_{story_index}"
            story_text = [sentence.replace(name, placeholder) for sentence in story_text]
            placeholders[placeholder] = name

            for i in range(len(new_story_tuple)-1):
                if new_story_tuple[i] == name:
                    new_story_tuple[i] = placeholder

        return story_text, tuple(new_story_tuple), placeholders
    
    @staticmethod
    def replace_placeholders_with_random_names(story, answer_tuple, names):
        # replace placeholders with random names ensuring balanced distribution
        used_names = Counter()
        name_mapping = {}

        for sentence in story:
            for placeholder in set(re.findall(r'PLACEHOLDER_[\w]+_\d+', sentence)):
                if placeholder not in name_mapping:
                    names_sorted_by_frequency = sorted(names, key=lambda name: used_names[name])
                    least_used_names = [name for name in names_sorted_by_frequency 
                                      if used_names[name] == used_names[names_sorted_by_frequency[0]]]
                    new_name = random.choice(least_used_names)
                    name_mapping[placeholder] = new_name
                    used_names[new_name] += 1

        processed_story = [sentence for sentence in story]
        for placeholder, new_name in name_mapping.items():
            processed_story = [sentence.replace(placeholder, new_name) for sentence in processed_story]

        processed_answer_tuple = tuple(name_mapping.get(name, name) for name in answer_tuple[:-1]) + (answer_tuple[-1],)

        return processed_story, processed_answer_tuple
    
    @staticmethod
    def process_stories_in_dataframe(df, names):
        # replace names in stories with random alternatives while preserving metadata
        processed_stories = []
        metadata_columns = {}
        
        print(f"Processing {len(df)} stories with name replacement")

        for index, row in df.iterrows():
            story = row['story']
            
            # handle both 2-element and 3-element tuple formats (earlier version of dataset)
            if isinstance(story, tuple):
                if len(story) >= 3:  # 3-element tuple with metadata
                    story_text, story_tuple, metadata = story
                else:  # 2-element tuple without metadata
                    story_text, story_tuple = story
                    metadata = {}
            else:
                print(f"Warning: Unexpected story format at index {index}: {type(story)}")
                continue

            story_with_placeholders, new_story_tuple, placeholders = StoryProcessor.replace_names_with_placeholders(
                story_text, story_tuple, names, index)
            
            processed_story_text, processed_story_tuple = StoryProcessor.replace_placeholders_with_random_names(
                story_with_placeholders, new_story_tuple, names)

            # preserve the metadata in the processed story
            if len(story) >= 3:
                processed_stories.append((processed_story_text, processed_story_tuple, metadata))
            else:
                processed_stories.append((processed_story_text, processed_story_tuple))
            
            # store metadata for this story
            for col in df.columns:
                if col != 'story':
                    if col not in metadata_columns:
                        metadata_columns[col] = []
                    metadata_columns[col].append(row[col])

        new_df = pd.DataFrame({'story': processed_stories})
        
        # add back all metadata columns
        for col, values in metadata_columns.items():
            new_df[col] = values
            
        if 'category' in new_df.columns:
            categories = new_df['category'].value_counts()
            print("Category distribution after name replacement:")
            for category, count in categories.items():
                print(f"  - {category}: {count} stories ({count/len(new_df)*100:.1f}%)")
        else:
            print("Warning: No 'category' column in processed DataFrame")
            
        return new_df
    
    @staticmethod
    def process_text(text):
        # extract story, question, and answer from text
        lines = text.split('\n')
        story = []
        question = []
        answer = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if '?' in line:
                try:
                    parts = line.split('?')
                    question_text = parts[0].strip()
                    answer = parts[1].strip()
                    question = re.sub(r'^\d+ ', '', question_text).split()
                except ValueError:
                    print("Error processing line:", line)
                    raise
            else:
                story_line = re.sub(r'^\d+ ', '', line)
                story.append(story_line)

        return story, question, answer
    
    @staticmethod
    def process_dataframe_for_qa(stories, metadata_df=None):
        # process stories for question answering while preserving category metadata
        print(f"DEBUG - Type of stories: {type(stories)}")
        print(f"DEBUG - Type of metadata_df: {type(metadata_df)}")
        
        if isinstance(stories, pd.DataFrame):
            print(f"DEBUG - stories DataFrame columns: {list(stories.columns)}")
        
        all_data = []
        metadata = []
        
        # safe check 
        if isinstance(stories, pd.DataFrame):
            print("Warning: stories is a DataFrame, but expected a list of tuples.")
            if 'story' in stories.columns:
                stories_list = stories['story'].tolist()
            else:
                print("Error: No 'story' column found in DataFrame. Cannot process.")
                return all_data, metadata
        elif stories is None:
            print("Warning: stories is None")
            return all_data, metadata
        else:
            stories_list = stories
        
        print(f"Processing {len(stories_list)} stories for QA with metadata preservation")
        
        # debug 
        if len(stories_list) > 0:
            print(f"First story type: {type(stories_list[0])}")
            if isinstance(stories_list[0], tuple):
                print(f"  Tuple length: {len(stories_list[0])}")
                for i, elem in enumerate(stories_list[0]):
                    print(f"  Element {i} type: {type(elem)}")
        
        has_category = False
        if metadata_df is not None:
            if isinstance(metadata_df, pd.DataFrame) and 'category' in metadata_df.columns:
                has_category = True
                categories = metadata_df['category'].unique()
                print(f"Found categories in metadata: {list(categories)}")
        
        for i, story_tuple in enumerate(stories_list):
            try:
                if not isinstance(story_tuple, tuple):
                    raise ValueError(f"Expected story_tuple to be a tuple, got {type(story_tuple)}")
                    
                # handle both 2-element and 3-element tuple formats
                if len(story_tuple) >= 3 and isinstance(story_tuple[2], dict):
                    sentences, name_tuple, story_metadata = story_tuple
                    name1, name2, boolean = name_tuple
                else:
                    sentences, name_tuple = story_tuple
                    name1, name2, boolean = name_tuple
                    story_metadata = {}
                
                # format story and question
                formatted_story = [f"{j + 1} {sentence.strip()}" for j, sentence in enumerate(sentences)]
                question = f"Is {name2} following {name1}?"
                answer = 'yes' if boolean else 'no'
                full_story = '\n'.join(formatted_story) + '\n' + question + " " + answer

                processed_story, processed_question, processed_answer = StoryProcessor.process_text(full_story)
                
                # process metadata for this story
                story_metadata_dict = dict(story_metadata) if story_metadata else {}
                
                # add metadata from DataFrame if not already in story metadata
                if metadata_df is not None and i < len(metadata_df):
                    for col in metadata_df.columns:
                        if col != 'story' and col not in story_metadata_dict:
                            try:
                                story_metadata_dict[col] = metadata_df.iloc[i][col]
                            except Exception as e:
                                print(f"Error extracting metadata column {col}: {str(e)}")
                
                qa_item = [processed_story, processed_question, processed_answer]
                all_data.append(qa_item)
                metadata.append(story_metadata_dict)
                    
            except Exception as e:
                print(f"Error processing story at index {i}: {str(e)}")
                all_data.append([["Error processing story"], ["Error"], "no"])
                metadata.append({'error': str(e)})

        print(f"Processed {len(all_data)} stories for QA")
        
        if has_category:
            preserved_categories = set()
            for meta in metadata:
                if 'category' in meta:
                    preserved_categories.add(meta['category'])
            
            print(f"Preserved categories in processed data: {preserved_categories}")
        
        return all_data, metadata