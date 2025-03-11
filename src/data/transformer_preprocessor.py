# preprocesses data for transformer models by building vocabulary and vectorizing stories
# sequence encoding with special tokens and attention masks

import re
import pickle
import torch
from torch.utils.data import Dataset
from src.data.abstract_preprocessor import AbstractDataPreprocessor

class TransformerDataPreprocessor(AbstractDataPreprocessor):
    """Data preprocessor for Transformer model."""
    
    def __init__(self, max_length):
        self.max_length = max_length
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<SEP>': 2, '<CLS>': 3}
        self.word_count = 4  # account for PAD, UNK, SEP, and CLS
    
    def build_vocab(self, data):
        """Build vocabulary from the data."""
        for story_tuple in data:
            story, _, _ = story_tuple
            for sentence in story:
                words = re.sub(r'[^\w\s]', '', sentence).lower().split()
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = self.word_count
                        self.word_count += 1

        # ensure 'yes' and 'no' are in vocabulary
        for word in ['no', 'yes']:
            if word not in self.vocab:
                self.vocab[word] = self.word_count
                self.word_count += 1
        
        return self
    
    def vectorize_stories(self, data):
        """Vectorize stories for Transformer model input."""
        X, attention_masks, Y = [], [], []
        
        for story, query, answer in data:
            x = []
            for sentence in story:
                words = re.sub(r'[^\w\s]', '', sentence).lower().split()
                x.extend([self.vocab.get(word, self.vocab['<UNK>']) for word in words])
            
            query_text = f"{query[0]} {query[1]}"
            xq = [self.vocab.get(word, self.vocab['<UNK>']) for word in re.sub(r'[^\w\s]', '', query_text).lower().split()]
            
            # combine story, question with special tokens
            encoded_sequence = [self.vocab['<CLS>']] + x + [self.vocab['<SEP>']] + xq 
            encoded_sequence = encoded_sequence[:self.max_length]
            padded_sequence = encoded_sequence + [self.vocab['<PAD>']] * (self.max_length - len(encoded_sequence))
            
            # create attention mask
            attention_mask = [1 if token_id != self.vocab['<PAD>'] else 0 for token_id in padded_sequence]
            
            # process answer
            y = 1 if answer == 'yes' else 0
            
            X.append(padded_sequence)
            attention_masks.append(attention_mask)  
            Y.append(y)
        
        # convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.long)
        attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.bool)
        Y_tensor = torch.tensor(Y, dtype=torch.float)
        
        return X_tensor, attention_masks_tensor, Y_tensor
    
    def save(self, file_path):
        """Save vocabulary to file."""
        with open(file_path, 'wb') as handle:
            pickle.dump(self.vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self
    
    def load(self, file_path):
        """Load vocabulary from file."""
        with open(file_path, 'rb') as handle:
            self.vocab = pickle.load(handle)
            self.word_count = max(self.vocab.values()) + 1
        return self

class StoryDataset(Dataset):
    """Dataset class for PyTorch data loaders."""
    
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }