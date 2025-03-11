# utility functions for data analysis and visualization
# helps examine story content, actor distributions, and dataset balance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def count_words_in_story(story_tuple):
    """Count the number of words in a story."""
    sentences, _ = story_tuple
    word_count = sum(len(sentence.split()) for sentence in sentences)
    return word_count

def calculate_actor_frequencies(df, names):
    """Calculate the frequency of each actor name in the stories."""
    actor_counts = {name: 0 for name in names}
    
    for _, row in df.iterrows():
        story_text, _ = row['story']
        story_text_str = " ".join(story_text)
        
        for name in names:
            actor_counts[name] += story_text_str.count(name)
    
    return actor_counts

def create_actor_presence_heatmap(df, names):
    """Create a heatmap showing actor presence across stories."""
    presence_matrix = []
    
    for _, row in df.iterrows():
        story_text, _ = row['story']
        story_text_str = " ".join(story_text)
        
        presence = [1 if name in story_text_str else 0 for name in names]
        presence_matrix.append(presence)
    
    presence_df = pd.DataFrame(presence_matrix, columns=names)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(presence_df.T, cmap='Blues', cbar_kws={'label': 'Present'})
    plt.xlabel('Story Index')
    plt.ylabel('Actor Name')
    plt.title('Actor Presence Across Stories')
    
    return plt.gcf()

def extract_metadata(df):
    """Extract and calculate metadata from the stories dataframe."""
    metadata = {
        'total_stories': len(df),
        'categories': df['category'].value_counts().to_dict(),
        'story_types': df['story_type'].value_counts().to_dict(),
        'noun_counts': df['num_nouns'].value_counts().to_dict(),
        'sentence_counts': df['num_sentences'].value_counts().to_dict(),
        'avg_nouns_per_story': df['num_nouns'].mean(),
        'avg_sentences_per_story': df['num_sentences'].mean()
    }
    
    word_counts = []
    for _, row in df.iterrows():
        word_counts.append(count_words_in_story(row['story']))
    
    metadata['avg_words_per_story'] = np.mean(word_counts)
    metadata['min_words'] = min(word_counts)
    metadata['max_words'] = max(word_counts)
    
    return metadata

def check_data_balance(df):
    """Check the balance of the dataset across different attributes."""
    balance_info = {}
    
    # check balance of story types (pos/neg)
    story_type_counts = df['story_type'].value_counts()
    balance_info['story_type'] = {
        'counts': story_type_counts.to_dict(),
        'ratio': story_type_counts.iloc[0] / story_type_counts.iloc[-1] if len(story_type_counts) > 1 else 1.0
    }
    
    # check balance across number of nouns
    noun_counts = df['num_nouns'].value_counts()
    balance_info['num_nouns'] = {
        'counts': noun_counts.to_dict(),
        'distribution': (noun_counts / len(df)).to_dict()
    }
    
    # check balance across categories
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        balance_info['category'] = {
            'counts': category_counts.to_dict(),
            'distribution': (category_counts / len(df)).to_dict()
        }
    
    # check balance of story types within each category
    if 'category' in df.columns:
        cross_tab = pd.crosstab(df['category'], df['story_type'])
        balance_info['category_by_story_type'] = cross_tab.to_dict()
    
    return balance_info