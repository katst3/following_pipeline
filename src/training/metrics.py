# provides metrics calculation functions for evaluating LSTM and transformer models

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

class MetricsProcessor:
    """Process and calculate metrics for model evaluation."""
    
    @staticmethod
    def calculate_accuracy_by_category(model, inputs, category_indices, framework='lstm'):
        """Calculate accuracy for different data categories."""
        accuracies = {}
        
        for category, indices in category_indices.items():
            if not indices:
                continue
                
            if framework == 'lstm':
                if isinstance(inputs, tuple) and len(inputs) == 3:
                    X, Xq, Y = inputs
                    X_cat, Xq_cat, Y_cat = X[indices], Xq[indices], Y[indices]
                    _, accuracy = model.evaluate(X_cat, Xq_cat, answers_test=Y_cat)                
                else:
                    raise ValueError("Expected inputs to be a tuple of (X, Xq, Y) for lstm")
            
            elif framework == 'transformer':
                if isinstance(inputs, tuple) and len(inputs) == 3:
                    X, masks, Y = inputs
                    X_cat = X[indices]
                    masks_cat = masks[indices]
                    Y_cat = Y[indices]
                    
                    # set model to evaluation mode
                    model.eval()
                    with torch.no_grad():
                        outputs = model(X_cat, masks_cat)
                        predictions = (torch.sigmoid(outputs) > 0.5).float()
                        accuracy = (predictions == Y_cat).float().mean().item()
                else:
                    raise ValueError("Expected inputs to be a tuple of (X, masks, Y) for transformer")
            
            else:
                raise ValueError(f"Unsupported framework: {framework}")
            
            accuracies[category] = accuracy
            
        return accuracies
    
    @staticmethod
    def calculate_confusion_matrix(model, inputs, framework='lstm'):
        """Calculate confusion matrix for model predictions."""
        if framework == 'lstm':
            if isinstance(inputs, tuple) and len(inputs) == 3:
                X, Xq, Y = inputs
                y_pred = (model.predict(X, Xq) > 0.5).astype(int).flatten()                
                y_true = Y
            else:
                raise ValueError("Expected inputs to be a tuple of (X, Xq, Y) for LSTM")
        
        elif framework == 'transformer':
            if isinstance(inputs, tuple) and len(inputs) == 3:
                X, masks, Y = inputs
                
                # set model to evaluation mode
                model.eval()
                with torch.no_grad():
                    outputs = model(X, masks)
                    y_pred = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
                    y_true = Y.int().cpu().numpy()
            else:
                raise ValueError("Expected inputs to be a tuple of (X, masks, Y) for transformer")
        
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def analyze_errors(model, inputs, story_data, framework='lstm'):
        """Analyze prediction errors to understand model failures."""
        if framework == 'lstm':
            if isinstance(inputs, tuple) and len(inputs) == 3:
                X, Xq, Y = inputs
                y_pred = (model.predict(X, Xq) > 0.5).astype(int).flatten()
            else:
                raise ValueError("Expected inputs to be a tuple of (X, Xq, Y) for lstm")
        
        elif framework == 'transformer':
            if isinstance(inputs, tuple) and len(inputs) == 3:
                X, masks, Y = inputs
                
                # set model to evaluation mode
                model.eval()
                with torch.no_grad():
                    outputs = model(X, masks)
                    y_pred = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
                    Y = Y.int().cpu().numpy()
            else:
                raise ValueError("Expected inputs to be a tuple of (X, masks, Y) for transformer")
        
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        error_indices = np.where(y_pred != Y)[0]
        
        error_analysis = []
        for idx in error_indices:
            story_info = story_data.iloc[idx]
            story_text, story_tuple = story_info['story']
            
            error_info = {
                'index': idx,
                'story_text': story_text,
                'question': f"Is {story_tuple[1]} following {story_tuple[0]}?",
                'true_answer': 'yes' if story_tuple[2] else 'no',
                'predicted_answer': 'yes' if y_pred[idx] == 1 else 'no',
                'num_nouns': story_info['num_nouns'],
                'num_sentences': story_info['num_sentences'],
                'story_type': story_info['story_type'],
                'category': story_info['category']
            }
            
            error_analysis.append(error_info)
        
        return error_analysis