# handles training of LSTM models with progress tracking and model checkpointing

import heapq
import os
import re
import numpy as np
import traceback
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import load_model
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from src.training.model_trainer import ModelTrainer
from src.training.lstm_neptune_logger import LSTMNeptuneLogger
from src.models.lstm_model import QA_Model
import pandas as pd

class LSTMTrainer(ModelTrainer):
    """Training implementation for LSTM models"""
    
    def __init__(self, experiment_tracker, save_directory):
        """Initialize the LSTM trainer."""
        self.experiment_tracker = experiment_tracker
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)
    
    def train(self, model, train_data, val_data, metadata_train, metadata_val, hyperparams, epochs=100):
        """Train the LSTM model with progress bar and detailed logging."""
        # unpack data
        X_train, Xq_train, Y_train = train_data
        X_val, Xq_val, Y_val = val_data
        
        dataset_name = "model"
        
        try:
            print(f"DEBUG - metadata_train type: {type(metadata_train)}")
            
            if isinstance(metadata_train, dict) and 'dataset_name' in metadata_train:
                if hasattr(metadata_train['dataset_name'], 'iloc'):
                    dataset_name = str(metadata_train['dataset_name'].iloc[0])
                else:
                    dataset_name = str(metadata_train['dataset_name'])
            elif hasattr(metadata_train, 'dataset_name'):
                if hasattr(metadata_train.dataset_name, 'iloc'):
                    dataset_name = str(metadata_train.dataset_name.iloc[0])
                else:
                    dataset_name = str(metadata_train.dataset_name)
            elif hasattr(metadata_train, 'get'):
                temp_name = metadata_train.get('dataset_name', 'model')
                if hasattr(temp_name, 'iloc'):
                    dataset_name = str(temp_name.iloc[0])
                else:
                    dataset_name = str(temp_name)
            elif hasattr(metadata_train, 'columns') and 'dataset_name' in metadata_train.columns:
                if len(metadata_train['dataset_name']) > 0:
                    dataset_name = str(metadata_train['dataset_name'].iloc[0])
        except Exception as e:
            print(f"Error extracting dataset name: {str(e)}")
            dataset_name = "model"  # fallback
        
        dataset_name = re.sub(r'[^\w\-]', '_', str(dataset_name))
        
        if not dataset_name:
            dataset_name = "model"
            
        print(f"Using sanitized dataset name '{dataset_name}' for model saving")
        
        print("Creating LSTMNeptuneLogger for detailed metrics tracking")
        neptune_logger = LSTMNeptuneLogger(
            self.experiment_tracker,
            model.model,
            train_data,
            val_data,
            metadata_train,
            metadata_val
        )
        
        # create raw model directory
        run_id = self.experiment_tracker.run['sys/id'].fetch()
        raw_model_dir = os.path.join(self.save_directory, "raw")
        os.makedirs(raw_model_dir, exist_ok=True)
        
        # create a safe model path: !!! remove "weights" if save_weights=False   
        model_path = f"{raw_model_dir}/{dataset_name}_lstm_model_{run_id}_epoch_{{epoch:02d}}.weights.h5"  
        print(f"Model will be saved to pattern: {model_path}")
        
        checkpoint_callback = ModelCheckpoint(
            filepath=model_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
            save_freq='epoch'
        )
        
        tqdm_callback = TqdmCallback(verbose=0)
        
        best_val_acc = 0.0
        best_epoch = -1
        top_epochs = []
        
        history_train_loss = []
        history_train_acc = []
        history_val_loss = []
        history_val_acc = []
        
        print(f"\nStarting LSTM training for {epochs} epochs:")
        
        epoch_progress = tqdm(range(epochs), desc="Training", unit="epoch")
        
        # train the model with callbacks
        try:
            history = model.model.fit(
                [X_train, Xq_train], Y_train,
                validation_data=([X_val, Xq_val], Y_val),
                batch_size=hyperparams.get('batch_size', 32),
                epochs=epochs,
                callbacks=[neptune_logger, checkpoint_callback, tqdm_callback],
                verbose=0
            )
            
            for epoch in range(epochs):
                train_loss = history.history['loss'][epoch] if epoch < len(history.history['loss']) else None
                train_acc = history.history['accuracy'][epoch] if epoch < len(history.history['accuracy']) else None
                val_loss = history.history['val_loss'][epoch] if epoch < len(history.history['val_loss']) else None
                val_acc = history.history['val_accuracy'][epoch] if epoch < len(history.history['val_accuracy']) else None
                
                if train_loss and train_acc and val_loss and val_acc:
                    epoch_progress.set_postfix({
                        'train_loss': f"{train_loss:.4f}",
                        'train_acc': f"{train_acc:.4f}",
                        'val_loss': f"{val_loss:.4f}",
                        'val_acc': f"{val_acc:.4f}"
                    })
                
                # track best epoch
                if val_acc and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    
                    # update epoch progress with best indicator
                    epoch_progress.set_postfix({
                        'train_loss': f"{train_loss:.4f}",
                        'train_acc': f"{train_acc:.4f}",
                        'val_loss': f"{val_loss:.4f}",
                        'val_acc': f"{val_acc:.4f}",
                        'best': f"✓ (epoch {epoch})"
                    })
                    
                # track top 3 epochs
                if val_acc:
                    if len(top_epochs) < 3:
                        heapq.heappush(top_epochs, (val_acc, epoch))
                    else:
                        heapq.heappushpop(top_epochs, (val_acc, epoch))
                
                if train_loss: history_train_loss.append(train_loss)
                if train_acc: history_train_acc.append(train_acc)
                if val_loss: history_val_loss.append(val_loss)
                if val_acc: history_val_acc.append(val_acc)
                
                epoch_progress.update(1)
                
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            traceback.print_exc()
            
            # try with a fallback model path
            fallback_path = f"{raw_model_dir}/lstm_model_{{epoch:02d}}.weights.h5"
            print(f"Trying fallback path pattern: {fallback_path}")
            
            checkpoint_callback = ModelCheckpoint(
                filepath=fallback_path,
                save_weights_only=True,
                save_best_only=False,
                verbose=0,
                save_freq='epoch'
            )
            
            # try training again with fallback path
            history = model.model.fit(
                [X_train, Xq_train], Y_train,
                validation_data=([X_val, Xq_val], Y_val),
                batch_size=hyperparams.get('batch_size', 32),
                epochs=epochs,
                callbacks=[neptune_logger, checkpoint_callback, tqdm_callback],
                verbose=0
            )
        
        # log final summary metrics
        print("Training completed. Calling on_train_end for final processing...")
        neptune_logger.on_train_end()
        
        # log top epochs directly - removing Neptune logging but keeping console output
        top_epochs_sorted = sorted(top_epochs, reverse=True)
        print(f"\n{'='*60}")
        print(f"FINAL TOP 3 EPOCHS SUMMARY:")
        
        
        for i, (acc, epoch) in enumerate(top_epochs_sorted):
            rank = i + 1
            print(f"  #{rank}: Epoch {epoch} - Validation Accuracy: {acc:.4f}")
            

            model_file = model_path.format(epoch=epoch)
            if os.path.exists(model_file):
                print(f"  - Model file saved at: {model_file}")
            else:
                # try fallback path
                fallback_file = f"{raw_model_dir}/lstm_model_{epoch:02d}.weights.h5"
                if os.path.exists(fallback_file):
                    print(f"  - Model file saved at: {fallback_file}")
        
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        
        return {
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'top_epochs': [e for _, e in top_epochs_sorted],
            'history': {
                'train_loss': history_train_loss,
                'train_accuracy': history_train_acc,
                'val_loss': history_val_loss,
                'val_accuracy': history_val_acc
            }
        }
    
    def save_model(self, model, path):
        """Save LSTM model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if not isinstance(path, str):
            path = str(path)
            
        path = re.sub(r'[^\w\-./\\]', '_', path)
        
        try:
            model.model.save_weights(path)
            print(f"Saved LSTM model weights to {path}")
            return path
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            traceback.print_exc()
            fallback_path = os.path.join(os.path.dirname(path), f"model_{id(model)}.weights.h5")
            try:
                model.model.save_weights(fallback_path)
                print(f"Saved LSTM model weights to fallback path: {fallback_path}")
                return fallback_path
            except Exception as e2:
                print(f"Error saving model to fallback path: {str(e2)}")
                traceback.print_exc()
                return None
    
    def load_model(self, model_config, path):
        """Load LSTM model from disk."""
        # create a new QA_Model instance
        qa_model = QA_Model(
            vocab_size=model_config['vocab_size'],
            max_story_len=model_config.get('max_story_len', 450),
            max_question_len=model_config.get('max_question_len', 5),
            hyperparams=model_config.get('hyperparams', {})
        )
        
        # load weights from the saved model
        try:
            qa_model.model.load_weights(path)
            print(f"Successfully loaded LSTM model weights from {path}")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            traceback.print_exc()
            # try loading as full model if loading weights fails
            try:
                qa_model.model = load_model(path)
                print(f"Successfully loaded full LSTM model from {path}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {str(e2)}")
        
        return qa_model
        
    def _find_column_by_patterns(self, dataframe, patterns, default_value=None):
        """Find a column in a dataframe that matches any of the provided patterns."""
        import re
        
        if not hasattr(dataframe, 'columns'):
            print(f"Warning: DataFrame doesn't have columns attribute")
            return default_value
            
        print(f"Available columns: {list(dataframe.columns)}")
        print(f"Looking for patterns: {patterns}")
        
        for pattern in patterns:
            if pattern in dataframe.columns:
                print(f"Found exact column match: '{pattern}'")
                return pattern
                
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = [col for col in dataframe.columns if regex.search(col)]
            if matches:
                print(f"Found regex match for '{pattern}': '{matches[0]}'")
                return matches[0]
                
        print(f"Warning: No column found matching any of these patterns: {patterns}")
        return default_value

    def evaluate(self, model, eval_data, metadata_eval, eval_set_name="validation"):
        """Evaluate the LSTM model with detailed metrics logging."""
        X_eval, Xq_eval, Y_eval = eval_data
        
        print(f"\nEvaluating LSTM model on {eval_set_name} set:")
        loss, accuracy = model.model.evaluate([X_eval, Xq_eval], Y_eval)
        
        self.experiment_tracker.log_metric(f"avg_metrics/{eval_set_name}_loss", loss)
        self.experiment_tracker.log_metric(f"avg_metrics/{eval_set_name}_accuracy", accuracy)
        
        detailed_metrics = {}
        
        is_df_metadata = hasattr(metadata_eval, 'columns')
        
        if is_df_metadata:
            print(f"Dataset '{eval_set_name}' metadata columns: {list(metadata_eval.columns)}")
            
            # find required columns
            category_col = None
            for col in ['category', 'categories', 'class', 'type']:
                if col in metadata_eval.columns:
                    category_col = col
                    break
                    
            noun_count_col = None
            for col in ['num_actors', 'n_actors', 'num_nouns', 'noun_count', 'actor_count']:
                if col in metadata_eval.columns:
                    noun_count_col = col
                    break
            
            if not category_col or not noun_count_col:
                print(f"WARNING: Cannot find required columns in metadata. Found: {list(metadata_eval.columns)}")
                categories = ['default_category']
                noun_counts = [0]
            else:
                print(f"Using category column: '{category_col}' and noun count column: '{noun_count_col}'")
                categories = sorted(metadata_eval[category_col].unique())
                
                try:
                    metadata_eval[noun_count_col] = pd.to_numeric(metadata_eval[noun_count_col], errors='coerce')
                    noun_counts = sorted(metadata_eval[noun_count_col].dropna().unique())
                except Exception as e:
                    print(f"Error processing noun counts: {str(e)}")
                    noun_counts = [0]
        else:
            # handle dictionary metadata format
            print(f"Using dictionary metadata format. Keys: {list(metadata_eval.keys()) if isinstance(metadata_eval, dict) else 'unknown'}")
            categories = ['default_category']
            noun_counts = [0]
            # extract categories and noun counts from the dictionary if possible
            if isinstance(metadata_eval, dict) and 'categories' in metadata_eval:
                categories = metadata_eval['categories']
            elif isinstance(metadata_eval, dict) and 'category' in metadata_eval:
                categories = [metadata_eval['category']]
            elif isinstance(metadata_eval, dict) and 'category_counts' in metadata_eval:
                categories = list(metadata_eval['category_counts'].keys())
                
            if isinstance(metadata_eval, dict) and 'num_actors' in metadata_eval:
                noun_counts = [metadata_eval['num_actors']]
            elif isinstance(metadata_eval, dict) and 'avg_num_actors' in metadata_eval:
                noun_counts = [int(metadata_eval['avg_num_actors'])]
            elif isinstance(metadata_eval, dict) and 'n_actors' in metadata_eval:
                noun_counts = [metadata_eval['n_actors']]
            elif isinstance(metadata_eval, dict) and 'num_nouns' in metadata_eval:
                noun_counts = [metadata_eval['num_nouns']]
        
        print(f"Evaluating metrics for {len(categories)} categories and {len(noun_counts)} noun counts")
        
        predictions = model.model.predict([X_eval, Xq_eval])
        y_pred = (predictions > 0.5).astype(int).flatten()
        
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS BY CATEGORY:")
        
        # calculate metrics for each category and noun count combination
        for category in categories:
            detailed_metrics[category] = {}
            
            for noun_count in noun_counts:
                try:
                    if is_df_metadata and category_col and noun_count_col:
                        indices = metadata_eval[
                            (metadata_eval[category_col] == category) & 
                            (metadata_eval[noun_count_col] == noun_count)
                        ].index.values
                        
                        if len(indices) == 0:
                            print(f"  - {category}/noun_{noun_count}: No samples found")
                            continue
                        
                        valid_indices = [i for i in indices if i < len(Y_eval)]
                        
                        if len(valid_indices) > 0:
                            correct = sum(y_pred[valid_indices] == Y_eval[valid_indices])
                            total = len(valid_indices)
                            eval_acc = correct / total
                            
                            detailed_metrics[category][noun_count] = eval_acc
                            metric_key = f"detailed_metrics/{eval_set_name}/{category}/noun_{noun_count}"
                            self.experiment_tracker.log_metric(metric_key, eval_acc)
                            
                            print(f"  - {category}/noun_{noun_count}: Accuracy={eval_acc:.4f} (from {total} samples)")
                        else:
                            print(f"  - {category}/noun_{noun_count}: No valid indices found")
                    else:
                        detailed_metrics[category][noun_count] = accuracy
                        print(f"  - {category}/noun_{noun_count}: Using overall accuracy={accuracy:.4f}")
                except Exception as e:
                    print(f"Error calculating metrics for {category}, noun_count={noun_count}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    detailed_metrics[category][noun_count] = 0.0
        
        print(f"{'='*50}\n")
        
        sample_counts = {}
        if is_df_metadata and category_col and noun_count_col:
            for cat in categories:
                sample_counts[cat] = {}
                for n in noun_counts:
                    count = len(metadata_eval[(metadata_eval[category_col] == cat) & 
                                            (metadata_eval[noun_count_col] == n)])
                    sample_counts[cat][n] = count
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_values': Y_eval,
            'detailed_metrics': detailed_metrics,
            'sample_counts': sample_counts
        }