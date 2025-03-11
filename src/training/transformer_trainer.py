# handles training of transformer models with progress tracking and model checkpointing

import os
import torch
import heapq
import re
import traceback
from tqdm.auto import tqdm
from src.training.model_trainer import ModelTrainer
from src.training.transformer_neptune_logger import TransformerNeptuneLogger
import pandas as pd
import numpy as np
class TransformerTrainer(ModelTrainer):
    """Training implementation for transformer models."""
    
    def __init__(self, experiment_tracker, save_directory):
        """Initialize the transformer trainer."""
        self.experiment_tracker = experiment_tracker
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)
    
    def train(self, model, train_data, val_data, metadata_train, metadata_val, hyperparams, epochs=100):
        """Train the transformer model with progress bar."""
        # unpack data
        X_train, mask_train, Y_train = train_data
        X_val, mask_val, Y_val = val_data
        
        dataset_name = "model"
        
        # try to extract from metadata, heavy sanitizing 
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
            dataset_name = "model"  

        dataset_name = re.sub(r'[^\w\-]', '_', str(dataset_name))
        
        if not dataset_name:
            dataset_name = "model"
            
        print(f"Using sanitized dataset name '{dataset_name}' for model saving")
        
        # create DataLoaders
        batch_size = hyperparams.get('batch_size', 32)
        
        # convert to Dataset objects
        train_dataset = torch.utils.data.TensorDataset(X_train, mask_train, Y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, mask_val, Y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # neptune logger
        print("Creating TransformerNeptuneLogger for detailed metrics tracking")
        neptune_logger = TransformerNeptuneLogger(
            self.experiment_tracker,
            model,
            train_data,
            val_data,
            metadata_train,
            metadata_val
        )
        
        # optimizer
        learning_rate = hyperparams.get('learning_rate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # loss function
        criterion = torch.nn.BCEWithLogitsLoss()
        
        best_val_acc = 0.0
        best_model_state = None
        best_epoch = -1
        top_epochs = []
        
        # training loop
        run_id = self.experiment_tracker.run['sys/id'].fetch()
        
        history_train_loss = []
        history_train_acc = []
        history_val_loss = []
        history_val_acc = []
        
        raw_model_dir = os.path.join(self.save_directory, "raw")
        os.makedirs(raw_model_dir, exist_ok=True)
        
        # create a safe model path
        model_path = f"{raw_model_dir}/{dataset_name}_transformer_model_{run_id}_epoch_{{epoch:02d}}.pt"
        print(f"Model will be saved to pattern: {model_path}")
        
        print(f"\nStarting transformer training for {epochs} epochs:")
        
        epoch_progress = tqdm(range(epochs), desc="Training", unit="epoch")
        
        # update model saving paths 
        for epoch in epoch_progress:
            model.train()
            epoch_train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_batches = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                                leave=False, unit="batch")
            
            for inputs, masks, labels in train_batches:
                optimizer.zero_grad()
                
                # forward pass
                outputs = model(inputs, masks)
                
                # ensure outputs and labels have compatible shapes
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)  # shape [batch_size]
                
                # calculate loss
                loss = criterion(outputs, labels)
                
                # backward pass
                loss.backward()
                optimizer.step()
                
                # track statistics
                epoch_train_loss += loss.item() * inputs.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                # update training batch progress bar
                current_loss = loss.item()
                current_acc = (predictions == labels).float().mean().item()
                train_batches.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'acc': f"{current_acc:.4f}"
                })
            
            # calculate training metrics
            epoch_train_loss = epoch_train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # validation phase
            model.eval()
            epoch_val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # create progress bar for validation batches
            val_batches = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", 
                              leave=False, unit="batch")
            
            with torch.no_grad():
                for inputs, masks, labels in val_batches:
                    # forward pass
                    outputs = model(inputs, masks)
                    
                    # ensure outputs and labels have compatible shapes
                    if outputs.dim() > 1 and outputs.size(1) == 1:
                        outputs = outputs.squeeze(1)
                    
                    # calculate loss
                    loss = criterion(outputs, labels)
                    
                    # track statistics
                    epoch_val_loss += loss.item() * inputs.size(0)
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
                    
                    # update validation batch progress bar
                    current_loss = loss.item()
                    current_acc = (predictions == labels).float().mean().item()
                    val_batches.set_postfix({
                        'loss': f"{current_loss:.4f}",
                        'acc': f"{current_acc:.4f}"
                    })
            
            # calculate validation metrics
            epoch_val_loss = epoch_val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            history_train_loss.append(epoch_train_loss)
            history_train_acc.append(epoch_train_acc)
            history_val_loss.append(epoch_val_loss)
            history_val_acc.append(epoch_val_acc)
            
            epoch_progress.set_postfix({
                'train_loss': f"{epoch_train_loss:.4f}",
                'train_acc': f"{epoch_train_acc:.4f}",
                'val_loss': f"{epoch_val_loss:.4f}",
                'val_acc': f"{epoch_val_acc:.4f}"
            })
            
            # Log epoch metrics through the Neptune logger
            neptune_logger.on_epoch_end(epoch, {
                'train_loss': epoch_train_loss,
                'train_acc': epoch_train_acc,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc
            })
            
            # save model if it's the best so far
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                
                current_model_path = model_path.format(epoch=epoch)
                
                print(f"\n{'='*60}")
                print(f"NEW BEST MODEL: Saving to: {current_model_path}")
                print(f"{'='*60}\n")
                
                try:
                    torch.save(model.state_dict(), current_model_path)
                except Exception as e:
                    print(f"Error saving model: {str(e)}")
                    try:
                        fallback_path = f"{raw_model_dir}/model_{run_id}_epoch_{epoch:02d}.pt"
                        print(f"Trying fallback path: {fallback_path}")
                        torch.save(model.state_dict(), fallback_path)
                    except Exception as e2:
                        print(f"Error saving model with fallback path: {str(e2)}")
                
                epoch_progress.set_postfix({
                    'train_loss': f"{epoch_train_loss:.4f}",
                    'train_acc': f"{epoch_train_acc:.4f}",
                    'val_loss': f"{epoch_val_loss:.4f}",
                    'val_acc': f"{epoch_val_acc:.4f}",
                    'best': f"✓ (epoch {epoch})"
                })
            
            # track top 3 epochs
            if len(top_epochs) < 3:
                heapq.heappush(top_epochs, (epoch_val_acc, epoch))
            else:
                heapq.heappushpop(top_epochs, (epoch_val_acc, epoch))
                
            # Save model periodically
            if (epoch + 1) % 10 == 0:
                current_model_path = model_path.format(epoch=epoch)
                try:
                    torch.save(model.state_dict(), current_model_path)
                except Exception as e:
                    print(f"Error saving periodic model: {str(e)}")
                    # try fallback path
                    try:
                        fallback_path = f"{raw_model_dir}/model_{run_id}_epoch_{epoch:02d}.pt"
                        torch.save(model.state_dict(), fallback_path)
                    except Exception as e2:
                        print(f"Error saving periodic model with fallback path: {str(e2)}")
       
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print("Training completed. Calling on_train_end for final processing...")
        neptune_logger.on_train_end()
        
        # log top epochs directly - only keeping console output
        top_epochs_sorted = sorted(top_epochs, reverse=True)
        print(f"\n{'='*60}")
        print(f"FINAL TOP 3 EPOCHS SUMMARY:")
        
        for i, (acc, epoch) in enumerate(top_epochs_sorted):
            rank = i + 1
            print(f"  #{rank}: Epoch {epoch} - Validation Accuracy: {acc:.4f}")
            
            # Save model files but don't log to Neptune
            current_model_path = model_path.format(epoch=epoch)
            if os.path.exists(current_model_path):
                print(f"  - Model file saved at: {current_model_path}")
            else:
                # try fallback path
                fallback_file = f"{raw_model_dir}/model_{run_id}_epoch_{epoch:02d}.pt"
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
        """Save transformer model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if not isinstance(path, str):
            path = str(path)
            
        path = re.sub(r'[^\w\-./\\]', '_', path)
        
        try:
            torch.save(model.state_dict(), path)
            print(f"Saved transformer model state dictionary to {path}")
            return path
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            traceback.print_exc()
            fallback_path = os.path.join(os.path.dirname(path), f"model_{id(model)}.pt")
            try:
                torch.save(model.state_dict(), fallback_path)
                print(f"Saved transformer model state dictionary to fallback path: {fallback_path}")
                return fallback_path
            except Exception as e2:
                print(f"Error saving model to fallback path: {str(e2)}")
                traceback.print_exc()
                return None
    
    def load_model(self, model_config, path):
        """Load transformer model from disk.
        
        Args:
            model_config: The model configuration
            path: Path to the saved model state dictionary
            
        Returns:
            Model with loaded weights
        """
        try:
            # create a new model instance with the given configuration
            from src.models.transformer_model import TransformerModel
            model = TransformerModel(**model_config)
            
            # load state dictionary
            model.load_state_dict(torch.load(path))
            print(f"Successfully loaded transformer model state dictionary from {path}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load transformer model: {str(e)}")
    
    def _find_column_by_patterns(self, dataframe, patterns, default_value=None):
        """Find a column in a dataframe that matches any of the provided patterns."""
        import re
        
        if not hasattr(dataframe, 'columns'):
            print(f"Warning: DataFrame doesn't have columns attribute")
            return default_value
            
        # Print all available columns for debugging
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
        """Evaluate the transformer model with detailed metrics logging."""
        # unpack data
        X_eval, mask_eval, Y_eval = eval_data
        
        print(f"\nEvaluating transformer model on {eval_set_name} set:")
        
        model.eval()
        
        batch_size = 32  
        eval_dataset = torch.utils.data.TensorDataset(X_eval, mask_eval, Y_eval)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            sampler=torch.utils.data.SequentialSampler(eval_dataset)
        )
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        raw_outputs = np.zeros(len(Y_eval))
        y_pred = np.zeros(len(Y_eval))
        y_probs = np.zeros(len(Y_eval))
        sample_idx = 0
        
        with torch.no_grad():
            for inputs, masks, labels in tqdm(eval_loader, desc=f"Evaluating {eval_set_name}", unit="batch"):
                outputs = model(inputs, masks)
                
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
                batch_size = inputs.size(0)
                raw_outputs[sample_idx:sample_idx+batch_size] = outputs.cpu().numpy()
                
                probs = torch.sigmoid(outputs)
                y_probs[sample_idx:sample_idx+batch_size] = probs.cpu().numpy()
                
                predictions = (probs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                y_pred[sample_idx:sample_idx+batch_size] = predictions.cpu().numpy()
                sample_idx += batch_size
        
        print(f"Raw model outputs stats: min={raw_outputs.min():.4f}, max={raw_outputs.max():.4f}, mean={raw_outputs.mean():.4f}")
        print(f"Sigmoid probabilities stats: min={y_probs.min():.4f}, max={y_probs.max():.4f}, mean={y_probs.mean():.4f}")
        print(f"Predictions distribution: zeros={np.sum(y_pred == 0)}, ones={np.sum(y_pred == 1)}")
        
        if np.all(y_pred == y_pred[0]):
            print(f"WARNING: All predictions are the same value ({y_pred[0]})!")
            print("Using alternative approach for calculating metrics...")
            
            y_true = Y_eval.cpu().numpy()
            
            sorted_indices = np.argsort(-raw_outputs)  
            num_positives = np.sum(y_true)
            
            if num_positives > 0:
                y_pred = np.zeros_like(y_true)
                y_pred[sorted_indices[:int(num_positives)]] = 1
                
                correct = np.sum(y_pred == y_true)
                accuracy = correct / len(y_true)
                print(f"Alternative accuracy: {accuracy:.4f}")
            else:
                accuracy = correct / total
        else:
            accuracy = correct / total
        
        avg_loss = total_loss / total
        
        self.experiment_tracker.log_metric(f"avg_metrics/{eval_set_name}_loss", avg_loss)
        self.experiment_tracker.log_metric(f"avg_metrics/{eval_set_name}_accuracy", accuracy)
        
        detailed_metrics = {}
        
        is_df_metadata = hasattr(metadata_eval, 'columns')
        
        if is_df_metadata:
            print(f"Dataset '{eval_set_name}' metadata columns: {list(metadata_eval.columns)}")
            
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
                    import pandas as pd
                    metadata_eval[noun_count_col] = pd.to_numeric(metadata_eval[noun_count_col], errors='coerce')
                    noun_counts = sorted(metadata_eval[noun_count_col].dropna().unique())
                except Exception as e:
                    print(f"Error processing noun counts: {str(e)}")
                    noun_counts = [0]
        else:
            print(f"Using dictionary metadata format. Keys: {list(metadata_eval.keys()) if isinstance(metadata_eval, dict) else 'unknown'}")
            categories = ['default_category']
            noun_counts = [0]
            
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
        
        y_true = Y_eval.cpu().numpy()
        
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS BY CATEGORY:")
        
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
                        
                        valid_indices = [i for i in indices if i < len(y_true)]
                        
                        if len(valid_indices) > 0:
                            subset_true = y_true[valid_indices]
                            subset_raw = raw_outputs[valid_indices]
                            
                            num_positives = np.sum(subset_true)
                            
                            # use a percentile-based approach
                            if np.all(y_pred[valid_indices] == y_pred[valid_indices][0]) and num_positives > 0:
                                sorted_subset_indices = np.argsort(-subset_raw)
                                
                                subset_pred = np.zeros_like(subset_true)
                                subset_pred[sorted_subset_indices[:int(num_positives)]] = 1
                                
                                correct = np.sum(subset_pred == subset_true)
                                eval_acc = correct / len(subset_true)
                            else:
                                correct = sum(y_pred[valid_indices] == y_true[valid_indices])
                                eval_acc = correct / len(valid_indices)
                            
                            detailed_metrics[category][noun_count] = eval_acc
                            metric_key = f"detailed_metrics/{eval_set_name}/{category}/noun_{noun_count}"
                            self.experiment_tracker.log_metric(metric_key, eval_acc)
                            
                            print(f"  - {category}/noun_{noun_count}: Accuracy={eval_acc:.4f} (from {len(valid_indices)} samples)")
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
        
        print(f"Detailed metrics structure: {list(detailed_metrics.keys())}")
        for category in detailed_metrics:
            print(f"  - {category}: {list(detailed_metrics[category].keys())}")
        
        sample_counts = {}
        if is_df_metadata and category_col and noun_count_col:
            for cat in categories:
                sample_counts[cat] = {}
                for n in noun_counts:
                    count = len(metadata_eval[(metadata_eval[category_col] == cat) & 
                                            (metadata_eval[noun_count_col] == n)])
                    sample_counts[cat][n] = count
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_values': y_true,
            'detailed_metrics': detailed_metrics,
            'sample_counts': sample_counts
        }