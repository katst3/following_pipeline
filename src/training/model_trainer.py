
# abstract interface providing a common structure for lstm and transformer trainers 

import os
from abc import ABC, abstractmethod

class ModelTrainer(ABC):
    """Abstract base class for model training."""
    
    def __init__(self, experiment_tracker, save_directory="models"):
        self.experiment_tracker = experiment_tracker
        self.save_directory = save_directory
        
        # create save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
    @abstractmethod
    def train(self, model, train_data, val_data, metadata_train, metadata_val, hyperparams, epochs=100):
        """Train the model and return validation metrics."""
        pass
    
    @abstractmethod
    def save_model(self, model, path):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path):
        """Load model from disk."""
        pass