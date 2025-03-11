# preprocesses data for LSTM models by tokenizing and padding sequences
# handles separate story and question inputs

import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data.abstract_preprocessor import AbstractDataPreprocessor

class LSTMDataPreprocessor(AbstractDataPreprocessor):
    """Data preprocessor for LSTM model (TensorFlow)."""
    
    def __init__(self, max_vocab_size, max_story_len, max_question_len):
        self.max_vocab_size = max_vocab_size
        self.max_story_len = max_story_len
        self.max_question_len = max_question_len
        self.vocab = set(["<UNK>"])
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, 
                                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n', 
                                  oov_token="<UNK>")
    
    def build_vocab(self, data):
        """Build vocabulary from the data."""
        for story, question, _ in data:
            # extract unique words from story and question
            story_words = [word for sentence in story for word in sentence.split()]
            self.vocab = self.vocab.union(set(story_words))
            self.vocab = self.vocab.union(set(question))
        
        # add yes/no answers to vocabulary
        self.vocab.add('no')
        self.vocab.add('yes')
        
        # fit tokenizer on vocab
        self.tokenizer.fit_on_texts(list(self.vocab))
        
        return self
    
    def vectorize_stories(self, data):
        """Vectorize stories for LSTM model input."""
        X, Xq, Y = [], [], []
        
        for story, query, answer in data:
            # vectorize story
            x = [self.tokenizer.word_index.get(word.lower(), self.tokenizer.word_index["<UNK>"]) 
                for sentence in story 
                for word in sentence.split()]
            
            # vectorize question
            xq = [self.tokenizer.word_index.get(word.lower(), self.tokenizer.word_index["<UNK>"]) 
                 for word in query]
            
            # vectorize answer (binary)
            y = 1 if answer.strip().lower() == 'yes' else 0
            
            X.append(x)
            Xq.append(xq)
            Y.append(y)
        
        # pad sequences to fixed length
        X_padded = pad_sequences(X, maxlen=self.max_story_len)
        Xq_padded = pad_sequences(Xq, maxlen=self.max_question_len)
        
        return X_padded, Xq_padded, np.array(Y)
    
    def save(self, file_path):
        """Save tokenizer to file."""
        with open(file_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self
    
    def load(self, file_path):
        """Load tokenizer from file."""
        with open(file_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        return self