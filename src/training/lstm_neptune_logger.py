# detailed metrics Neptune logger for LSTM models  

import heapq
import numpy as np
import os
import traceback
from tensorflow.keras.callbacks import Callback

class LSTMNeptuneLogger(Callback):
    """Neptune logger for LSTM models with detailed tracking."""
    
    def __init__(self, experiment_tracker, model, train_data, val_data, metadata_train, metadata_val):
        super().__init__()
        self.tracker = experiment_tracker
        self._keras_model = model
        self.train_data = train_data  # (X_train, Xq_train, Y_train)
        self.val_data = val_data  # (X_val, Xq_val, Y_val)
        self.metadata_train = metadata_train  # DataFrame with metadata
        self.metadata_val = metadata_val  # DataFrame with metadata
        
        # print debugging info about metadata
        print(f"LSTMNeptuneLogger init:")
        print(f"  - Metadata train type: {type(self.metadata_train)}")
        print(f"  - Metadata val type: {type(self.metadata_val)}")
        
        # check if metadata contains required columns
        if hasattr(self.metadata_train, 'columns'):
            print(f"  - Metadata train columns: {list(self.metadata_train.columns)}")
            if 'category' in self.metadata_train.columns:
                print(f"  - Train categories: {self.metadata_train['category'].unique()}")
            else:
                print("  - No 'category' column found in training metadata")
                potential_cols = []
                for col in self.metadata_train.columns:
                    if 'cat' in col.lower() or 'type' in col.lower() or 'class' in col.lower() or 'group' in col.lower():
                        potential_cols.append(col)
                        print(f"  - Potential category column: '{col}' with values: {self.metadata_train[col].unique()}")
        
        if hasattr(self.metadata_val, 'columns'):
            print(f"  - Metadata val columns: {list(self.metadata_val.columns)}")
            if 'category' in self.metadata_val.columns:
                print(f"  - Val categories: {self.metadata_val['category'].unique()}")
        
        self.best_val_acc = 0
        self.best_epoch = -1
        self.top_epochs = []
        self.epoch_accuracies = {}  
        self.best_epoch_accuracies = {}
        
        # make sure metrics namespace is logged first by logging placeholders
        self.tracker.log_metric("avg_metrics/placeholder", 0)
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch during model training."""
        logs = logs or {}
        
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        print(f"Logging basic metrics for epoch {epoch}...")
        self.tracker.log_metric("avg_metrics/train_loss", train_loss, epoch)
        self.tracker.log_metric("avg_metrics/train_accuracy", train_acc, epoch)
        self.tracker.log_metric("avg_metrics/val_loss", val_loss, epoch)
        self.tracker.log_metric("avg_metrics/val_accuracy", val_acc, epoch)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch

            print(f"\n{'='*60}")
            print(f"NEW BEST EPOCH: {epoch} with validation accuracy: {val_acc:.4f}")
            print(f"{'='*60}\n")
        
        # update top epochs tracking (no logging)
        if len(self.top_epochs) < 3:
            heapq.heappush(self.top_epochs, (val_acc, epoch))
            top_epochs_sorted = sorted(self.top_epochs, reverse=True)
            self._print_top_epochs_info(top_epochs_sorted)  
        else:
            min_acc, min_epoch = min(self.top_epochs)
            
            old_top = sorted(self.top_epochs, reverse=True)
            heapq.heappushpop(self.top_epochs, (val_acc, epoch))
            new_top = sorted(self.top_epochs, reverse=True)
            
            if new_top != old_top:
                print(f"\n{'*'*60}")
                if val_acc > min_acc:
                    print(f"NEW TOP 3 EPOCH: {epoch} with val_acc {val_acc:.4f} replaced epoch {min_epoch} with val_acc {min_acc:.4f}")
                self._print_top_epochs_info(new_top)  
                print(f"{'*'*60}\n")
        
        # compute detailed accuracies but don't log them yet
        try:
            print(f"Computing detailed accuracies for epoch {epoch}...")
            self.epoch_accuracies[epoch] = self._compute_detailed_accuracies()
            
            if epoch == self.best_epoch:
                self.best_epoch_accuracies = self.epoch_accuracies.get(epoch, {})
        except Exception as e:
            print(f"Error in detailed metric computation: {str(e)}")
            traceback.print_exc()
        
        # save weights for top models, no logging
        if any(epoch == e[1] for e in self.top_epochs):
            model_filename = f"model_epoch_{epoch:02d}.weights.h5"
            self._keras_model.save_weights(model_filename)  

    def _find_column_by_patterns(self, dataframe, patterns, default_value=None):
        """Find a column in a dataframe that matches any of the provided patterns."""
        import re
        
        if not hasattr(dataframe, 'columns'):
            print(f"Warning: DataFrame doesn't have columns attribute")
            return default_value
        
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

    def _print_top_epochs_info(self, sorted_top_epochs):
        """Print information about top epochs without logging to Neptune."""
        print(f"CURRENT TOP 3 EPOCHS:")
        for i, (acc, epoch) in enumerate(sorted_top_epochs):
            rank = i + 1
            print(f"  #{rank}: Epoch {epoch} - Validation Accuracy: {acc:.4f}")

    def _compute_detailed_accuracies(self):
        """Compute detailed accuracies for category and noun count combinations."""
        X_train, Xq_train, Y_train = self.train_data
        X_val, Xq_val, Y_val = self.val_data
        
        accuracies = {}
        
        is_df_metadata = hasattr(self.metadata_val, 'columns')
        
        if is_df_metadata:
            category_col = self._find_column_by_patterns(self.metadata_val, ['category'], 'default_category')
            noun_count_col = self._find_column_by_patterns(self.metadata_val, ['num_actors', 'num_actors'], 'default_noun_count')
            
            if category_col != 'default_category':
                categories = self.metadata_val[category_col].unique()
            else:
                categories = ['default_category']
                
            if noun_count_col != 'default_noun_count':
                noun_counts = sorted(self.metadata_val[noun_count_col].unique())
            else:
                noun_counts = [0]
        else:
            print("Using dictionary metadata format")
            
            if 'categories' in self.metadata_val:
                categories = self.metadata_val['categories']
            elif 'category' in self.metadata_val:
                categories = [self.metadata_val['category']]
            elif 'category_counts' in self.metadata_val:
                categories = list(self.metadata_val['category_counts'].keys())
            else:
                categories = ['default_category']
                
            if 'num_actors' in self.metadata_val:
                noun_counts = [self.metadata_val['num_actors']]
            elif 'avg_num_actors' in self.metadata_val:
                noun_counts = [int(self.metadata_val['avg_num_actors'])]
            elif 'num_actors' in self.metadata_val:
                noun_counts = [self.metadata_val['num_actors']]
            else:
                noun_counts = [0]
        
        print(f"Using categories: {categories}")
        print(f"Using noun counts: {noun_counts}")
        
        # calculate overall and category-specific accuracies
        for category in categories:
            accuracies[category] = {}
            
            if is_df_metadata and category_col in self.metadata_train.columns:
                for noun_count in noun_counts:
                    try:
                        # filter by category and noun count
                        train_indices = self.metadata_train[
                            (self.metadata_train[category_col] == category) & 
                            (self.metadata_train[noun_count_col] == noun_count)
                        ].index.values
                        
                        val_indices = self.metadata_val[
                            (self.metadata_val[category_col] == category) & 
                            (self.metadata_val[noun_count_col] == noun_count)
                        ].index.values
                        
                        # ensure indices are valid
                        train_indices = [i for i in train_indices if i < len(Y_train)]
                        val_indices = [i for i in val_indices if i < len(Y_val)]
                        
                        if len(train_indices) > 0 and len(val_indices) > 0:
                            # calculate training accuracy for this category/noun count combo
                            train_acc = self._keras_model.evaluate(
                                [X_train[train_indices], Xq_train[train_indices]], 
                                Y_train[train_indices], 
                                verbose=0
                            )[1]
                            
                            # calculate validation accuracy for this category/noun count combo
                            val_acc = self._keras_model.evaluate(
                                [X_val[val_indices], Xq_val[val_indices]], 
                                Y_val[val_indices], 
                                verbose=0
                            )[1]
                            
                            accuracies[category][noun_count] = (train_acc, val_acc)
                        else:
                            # no data for this combination
                            accuracies[category][noun_count] = (0.0, 0.0)
                            
                    except Exception as e:
                        print(f"Error computing accuracy for {category}, noun_count={noun_count}: {e}")
                        traceback.print_exc()
                        accuracies[category][noun_count] = (0.0, 0.0)
            else:
                train_acc = self._keras_model.evaluate([X_train, Xq_train], Y_train, verbose=0)[1]
                val_acc = self._keras_model.evaluate([X_val, Xq_val], Y_val, verbose=0)[1]
                
                for noun_count in noun_counts:
                    accuracies[category][noun_count] = (train_acc, val_acc)
                    if not is_df_metadata:
                        break
        
        return accuracies
    
    def _log_detailed_metrics_as_single_values(self, epoch, rank):
        """Log detailed metrics for an epoch as single values."""
        if epoch in self.epoch_accuracies:
            print(f"Logging detailed metrics for top #{rank} epoch {epoch} as single values")
            metrics_logged = 0
            
            for category, noun_counts in self.epoch_accuracies[epoch].items():
                for noun_count, (train_acc, val_acc) in noun_counts.items():
                    self.tracker.log_metric(f"detailed_metrics/top{rank}_epoch{epoch}/{category}/noun_{noun_count}/train", train_acc)
                    self.tracker.log_metric(f"detailed_metrics/top{rank}_epoch{epoch}/{category}/noun_{noun_count}/val", val_acc)
                    
                    print(f"  - top{rank}_epoch{epoch}/{category}/noun_{noun_count}: Train={train_acc:.4f}, Val={val_acc:.4f}")
                    metrics_logged += 2
            
            print(f"  - Logged {metrics_logged} detailed metrics for top #{rank} epoch {epoch}")
    
    def on_train_end(self, logs=None):
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE - BEST RESULTS:")
        print(f"  - Best Epoch: {self.best_epoch}")
        print(f"  - Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        top_epochs_sorted = sorted(self.top_epochs, reverse=True)
        print(f"\n{'='*60}")
        print(f"TOP 3 EPOCHS SUMMARY:")
        for i, (acc, epoch) in enumerate(top_epochs_sorted):
            rank = i + 1
            print(f"  #{rank}: Epoch {epoch} - Validation Accuracy: {acc:.4f}")
            
            self._log_detailed_metrics_as_single_values(epoch, rank)
            
        print(f"{'='*60}\n")
        
        self._log_category_averages()

    def _log_category_averages(self):
        """Log average metrics by category from the best epoch as single values."""
        if not self.best_epoch_accuracies:
            print("No best epoch accuracies to summarize")
            return
            
        print(f"\n{'='*60}")
        print(f"CATEGORY AVERAGES FROM BEST EPOCH ({self.best_epoch}):")
        
        # calculate average accuracy by category
        for category, noun_counts in self.best_epoch_accuracies.items():
            train_accs = []
            val_accs = []
            
            for noun_count, (train_acc, val_acc) in noun_counts.items():
                train_accs.append(train_acc)
                val_accs.append(val_acc)
            
            if train_accs and val_accs:
                avg_train_acc = sum(train_accs) / len(train_accs)
                avg_val_acc = sum(val_accs) / len(val_accs)
                print(f"  - {category}: Train={avg_train_acc:.4f}, Val={avg_val_acc:.4f}")
                
                self.tracker.log_metric(f"detailed_metrics/category_averages/{category}/train", avg_train_acc)
                self.tracker.log_metric(f"detailed_metrics/category_averages/{category}/val", avg_val_acc)
        
        print(f"{'='*60}\n")

    def get_best_and_top_epochs(self):
        """Return information about best and top epochs."""
        top_epochs_sorted = sorted(self.top_epochs, reverse=True)
        return {
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'top_epochs': [(acc, epoch) for acc, epoch in top_epochs_sorted]
        }