# interface for consistent functionality for both data preprocessors

from abc import ABC, abstractmethod

class AbstractDataPreprocessor(ABC):
    """Abstract base class for data preprocessing."""
    
    @abstractmethod
    def build_vocab(self, data):
        """Build vocabulary from the data."""
        pass
    
    @abstractmethod
    def vectorize_stories(self, data):
        """Vectorize stories for model input."""
        pass
    
    @abstractmethod
    def save(self, file_path):
        """Save preprocessor state."""
        pass
    
    @abstractmethod
    def load(self, file_path):
        """Load preprocessor state."""
        pass