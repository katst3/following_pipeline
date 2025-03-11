# unified pipeline for the complete workflow for LSTM and transformer models
# handles data preparation, model creation, training, optimization, and evaluation

import os
import torch
import shutil
from datetime import datetime
import json
from tqdm.auto import tqdm
import pandas as pd
from debugging_functions import trace_evaluation_data_flow
from src.data.story_processor import StoryProcessor
from src.data.lstm_preprocessor import LSTMDataPreprocessor
from src.data.transformer_preprocessor import TransformerDataPreprocessor
from src.models.lstm_model import QA_Model
from src.models.transformer_model import TransformerModel
from src.training.lstm_trainer import LSTMTrainer
from src.training.transformer_trainer import TransformerTrainer
from src.optimization.optimizer import HyperparameterOptimizer

class UnifiedPipeline:
    
    def __init__(self, framework, experiment_tracker, model_save_dir="models"):
        """
        Initialize the pipeline.
        
        Args:
            framework: 'lstm' or 'transformer'
            experiment_tracker: ExperimentTracker instance
            model_save_dir: Directory to save models
        """
        self.framework = framework.lower()
        self.experiment_tracker = experiment_tracker
        self.model_save_dir = model_save_dir
        
        if self.framework == 'lstm':
            self.trainer = LSTMTrainer(experiment_tracker, model_save_dir)
            print(f"Initialized LSTMTrainer")
        elif self.framework == 'transformer':
            self.trainer = TransformerTrainer(experiment_tracker, model_save_dir)
            print(f"Initialized TransformerTrainer")
        else:
            raise ValueError(f"Unsupported framework: {framework}. Use 'lstm' or 'transformer'.")
        
    def prepare_data(self, dataset_name, stories_path, train_indices_path, valid_indices_path, 
                    validb_indices_path=None, test_indices_path=None, names=None, 
                    max_vocab_size=55, max_story_len=450, max_question_len=5, max_length=150):
        """
        Prepare data for model training.
        
        Args:
            dataset_name: Name of the dataset (e.g., '2directional', '4directional')
            stories_path: Path to the pickle file with stories
            train_indices_path: Path to JSON file with training indices
            valid_indices_path: Path to JSON file with validation indices
            validb_indices_path: Path to JSON file with secondary validation indices
            test_indices_path: Path to JSON file with test indices
            names: List of names to use for substitution
            max_vocab_size: Maximum vocabulary size for LSTM models
            max_story_len: Maximum story length for LSTM models
            max_question_len: Maximum question length for LSTM models
            max_length: Maximum sequence length for Transformer models
        
        Returns:
            Dictionary with preprocessed data
        """
        print(f"Step 1: Loading stories from {stories_path}")
        processor = StoryProcessor()
        dataset = processor.load_dataset(stories_path)
        print(f"  - Loaded dataset with {sum(len(v['pos']) + len(v['neg']) for k, v in dataset.items())} stories")
        
        # extract direction type from dataset_name
        direction_type = ""
        if "2dir" in dataset_name or "2directional" in dataset_name:
            direction_type = "2dir"
        elif "4dir" in dataset_name or "4directional" in dataset_name:
            direction_type = "4dir"
        else:
            direction_type = dataset_name  # fallback
        
        print(f"  - Detected direction type: {direction_type}")
        
        print(f"Step 2: Processing all tuple keys and mapping to density categories")
        merged_dataset = processor.filter_and_organize_stories(processor, dataset)        
        print(f"  - Mapped dataset has {len(merged_dataset)} stories")
        
        # add dataset_name to metadata in each story 
        print(f"Step 2b: Adding dataset_name to metadata")
        for item in merged_dataset:
            if 'story' in item:
                story = item['story']
                if isinstance(story, tuple) and len(story) > 2:
                    if isinstance(story[2], dict):
                        metadata = dict(story[2])
                        metadata['dataset_name'] = direction_type
                        new_story = (story[0], story[1], metadata)
                        item['story'] = new_story
                        
            item['dataset_name'] = direction_type
        
        # create DataFrame
        print(f"Step 3: Creating DataFrame")
        df_merged = processor.create_dataframe(merged_dataset)
        print(f"  - Created DataFrame with {len(df_merged)} rows")
        
        # ensure dataset_name column exists in DataFrame
        if 'dataset_name' not in df_merged.columns:
            print(f"  - Adding dataset_name column to DataFrame")
            df_merged['dataset_name'] = direction_type
        
        # print distribution of categories for verification
        if 'category' in df_merged.columns:
            print(f"  - Category distribution in DataFrame:")
            categories = df_merged['category'].value_counts()
            for category, count in categories.items():
                print(f"    * {category}: {count} stories ({count/len(df_merged)*100:.1f}%)")
        
        print(f"  - Dataset name values: {df_merged['dataset_name'].unique()}")
        
        # split into train and validation sets
        print(f"Step 4: Splitting data into train/validation/test sets")
        print(f"  - Loading train indices from {train_indices_path}")
        train_indices = processor.read_indices_from_json(train_indices_path)
        print(f"  - Loading validation indices from {valid_indices_path}")
        valid_indices = processor.read_indices_from_json(valid_indices_path)
        
        # optional additional evaluation sets
        validb_indices = None
        test_indices = None
        
        if validb_indices_path:
            print(f"  - Loading secondary validation indices from {validb_indices_path}")
            validb_indices = processor.read_indices_from_json(validb_indices_path)
        
        if test_indices_path:
            print(f"  - Loading test indices from {test_indices_path}")
            test_indices = processor.read_indices_from_json(test_indices_path)
        
        df_train = processor.split_dataframe(df_merged, train_indices)
        df_valid = processor.split_dataframe(df_merged, valid_indices)
        print(f"  - Split data: Train={len(df_train)}, Validation={len(df_valid)}")
        
        # create additional dataframes if indices are provided
        df_validb = None
        df_test = None
        
        if validb_indices:
            df_validb = processor.split_dataframe(df_merged, validb_indices)
            print(f"  - Secondary validation set: {len(df_validb)} examples")
        
        if test_indices:
            df_test = processor.split_dataframe(df_merged, test_indices)
            print(f"  - Test set: {len(df_test)} examples")
        
        # process stories with name replacement
        print(f"Step 5: Processing stories with name replacement")
        print(f"  - Processing train set...")
        unif_tr_df = processor.process_stories_in_dataframe(df_train, names)
        print(f"  - Processing validation set...")
        unif_val_df = processor.process_stories_in_dataframe(df_valid, names)

        unif_tr_df['dataset_name'] = direction_type
        unif_val_df['dataset_name'] = direction_type

        # process additional sets if available
        unif_valb_df = None
        unif_test_df = None

        if df_validb is not None:
            print(f"  - Processing secondary validation set...")
            unif_valb_df = processor.process_stories_in_dataframe(df_validb, names)
            unif_valb_df['dataset_name'] = direction_type

        if df_test is not None:
            print(f"  - Processing test set...")
            unif_test_df = processor.process_stories_in_dataframe(df_test, names)
            unif_test_df['dataset_name'] = direction_type
        
        print(f"Step 6: Transforming to QA format")
        print(f"  - Processing train data...")
        training_data, train_metadata = StoryProcessor.process_dataframe_for_qa(unif_tr_df['story'].tolist(), unif_tr_df)
        print(f"  - Processing validation data...")
        validation_data, val_metadata = StoryProcessor.process_dataframe_for_qa(unif_val_df['story'].tolist(), unif_val_df)
        
        validationb_data = None
        validb_metadata = None
        test_data = None
        test_metadata = None
        
        if unif_valb_df is not None:
            print(f"  - Processing secondary validation data...")
            validationb_data, validb_metadata = processor.process_dataframe_for_qa(unif_valb_df['story'], unif_valb_df)
        
        if unif_test_df is not None:
            print(f"  - Processing test data...")
            test_data, test_metadata = processor.process_dataframe_for_qa(unif_test_df['story'], unif_test_df)
        
        print(f"Step 7: Creating {self.framework} preprocessor")
        if self.framework == 'lstm':
            print(f"  - Creating LSTM preprocessor (max_vocab_size={max_vocab_size}, max_story_len={max_story_len}, max_question_len={max_question_len})")
            preprocessor = LSTMDataPreprocessor(max_vocab_size, max_story_len, max_question_len)
        else:  # transformer
            print(f"  - Creating Transformer preprocessor (max_length={max_length})")
            preprocessor = TransformerDataPreprocessor(max_length)
        
        # build vocabulary
        print(f"Step 8: Building vocabulary")
        preprocessor.build_vocab(training_data)
        if self.framework == 'lstm':
            print(f"  - Vocabulary size (capped): {max_vocab_size}")
        else:
            print(f"  - Vocabulary size: {len(preprocessor.vocab)}")
        
        # vectorize datasets
        print(f"Step 9: Vectorizing datasets")

        result = {
            'dataset_name': direction_type,
            'preprocessor': preprocessor,
            'train_df': unif_tr_df,
            'val_df': unif_val_df,
            'vocab_size': max_vocab_size if self.framework == 'lstm' else len(preprocessor.vocab)
        }
        
        # store metadata for each dataset
        result['train_metadata'] = train_metadata
        result['val_metadata'] = val_metadata
        
        if self.framework == 'lstm':
            print(f"  - Vectorizing training data...")
            inputs_train, queries_train, answers_train = preprocessor.vectorize_stories(training_data)
            print(f"    * Training data shapes: X={inputs_train.shape}, Xq={queries_train.shape}, Y={answers_train.shape}")
            
            print(f"  - Vectorizing validation data...")
            inputs_val, queries_val, answers_val = preprocessor.vectorize_stories(validation_data)
            print(f"    * Validation data shapes: X={inputs_val.shape}, Xq={queries_val.shape}, Y={answers_val.shape}")
            
            result['train_data'] = (inputs_train, queries_train, answers_train)
            result['val_data'] = (inputs_val, queries_val, answers_val)
            
            if validationb_data is not None:
                print(f"  - Vectorizing secondary validation data...")
                inputs_valb, queries_valb, answers_valb = preprocessor.vectorize_stories(validationb_data)
                print(f"    * Secondary validation data shapes: X={inputs_valb.shape}, Xq={queries_valb.shape}, Y={answers_valb.shape}")
                result['valb_data'] = (inputs_valb, queries_valb, answers_valb)
                result['valb_df'] = unif_valb_df
                result['valb_metadata'] = validb_metadata
            
            if test_data is not None:
                print(f"  - Vectorizing test data...")
                inputs_test, queries_test, answers_test = preprocessor.vectorize_stories(test_data)
                print(f"    * Test data shapes: X={inputs_test.shape}, Xq={queries_test.shape}, Y={answers_test.shape}")
                result['test_data'] = (inputs_test, queries_test, answers_test)
                result['test_df'] = unif_test_df
                result['test_metadata'] = test_metadata
                
        else:  # transformer
            print(f"  - Vectorizing training data...")
            inputs_train, attention_masks_train, answers_train = preprocessor.vectorize_stories(training_data)
            print(f"    * Training data shapes: X={inputs_train.shape}, masks={attention_masks_train.shape}, Y={answers_train.shape}")
            
            print(f"  - Vectorizing validation data...")
            inputs_val, attention_masks_val, answers_val = preprocessor.vectorize_stories(validation_data)
            print(f"    * Validation data shapes: X={inputs_val.shape}, masks={attention_masks_val.shape}, Y={answers_val.shape}")
            
            result['train_data'] = (inputs_train, attention_masks_train, answers_train)
            result['val_data'] = (inputs_val, attention_masks_val, answers_val)
            
            if validationb_data is not None:
                print(f"  - Vectorizing secondary validation data...")
                inputs_valb, attention_masks_valb, answers_valb = preprocessor.vectorize_stories(validationb_data)
                print(f"    * Secondary validation data shapes: X={inputs_valb.shape}, masks={attention_masks_valb.shape}, Y={answers_valb.shape}")
                result['valb_data'] = (inputs_valb, attention_masks_valb, answers_valb)
                result['valb_df'] = unif_valb_df
                result['valb_metadata'] = validb_metadata
            
            if test_data is not None:
                print(f"  - Vectorizing test data...")
                inputs_test, attention_masks_test, answers_test = preprocessor.vectorize_stories(test_data)
                print(f"    * Test data shapes: X={inputs_test.shape}, masks={attention_masks_test.shape}, Y={answers_test.shape}")
                result['test_data'] = (inputs_test, attention_masks_test, answers_test)
                result['test_df'] = unif_test_df
                result['test_metadata'] = test_metadata
        
        # save preprocessor 
        preprocessor_path = f"{self.model_save_dir}/preprocessor_{direction_type}_{self.framework}.pkl"
        print(f"Step 10: Saving preprocessor to {preprocessor_path}")
        preprocessor.save(preprocessor_path)
        
        print(f"Data preparation complete!")
        return result

        
    def create_model(self, data_info, hyperparams):
        """
        Create a model based on the framework.
        
        Args:
            data_info: Output from prepare_data
            hyperparams: Model hyperparameters
        
        Returns:
            Created model
        """
        dataset_name = data_info.get('dataset_name', '2directional')
        
        if self.framework == 'lstm':
            hidden_layers = hyperparams.get("hidden_layers", 74 if dataset_name == "2directional" else 134)
            
            print(f"Creating TensorFlow LSTM model:")
            print(f"  - Vocabulary size: {data_info['vocab_size']}")
            print(f"  - Max story length: {hyperparams.get('max_story_len', 450)}")
            print(f"  - Max question length: {hyperparams.get('max_question_len', 5)}")
            print(f"  - Hidden layers: {hidden_layers}")
            print(f"  - Dropout: {hyperparams.get('dropout', 0.39)}")
            
            model = QA_Model(
                vocab_size=data_info['vocab_size'],
                max_story_len=hyperparams.get('max_story_len', 450),
                max_question_len=hyperparams.get('max_question_len', 5),
                hyperparams=hyperparams
            )
            print(f"Model created successfully")
            return model
        else:  # transformer
            d_model = hyperparams.get('d_model', 256 if dataset_name == "2directional" else 512)
            d_hid = hyperparams.get('d_hid', 569 if dataset_name == "2directional" else 565)
            nlayers = hyperparams.get('nlayers', 4 if dataset_name == "2directional" else 3)
            nhead = hyperparams.get('nhead', 2 if dataset_name == "2directional" else 4)
            dropout = hyperparams.get('dropout', 0.413 if dataset_name == "2directional" else 0.565)
            
            print(f"Creating PyTorch Transformer model:")
            print(f"  - Vocabulary size: {data_info['vocab_size']}")
            print(f"  - Model dimension (d_model): {d_model}")
            print(f"  - Hidden dimension (d_hid): {d_hid}")
            print(f"  - Number of layers: {nlayers}")
            print(f"  - Number of attention heads: {nhead}")
            print(f"  - Dropout: {dropout}")
            
            model = TransformerModel(
                ntoken=data_info['vocab_size'],
                d_model=d_model,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=nlayers,
                dropout=dropout
            )
            print(f"Model created successfully")
            return model
    
    def load_model_from_path(self, model_path, dataset_name, vocab_size, hyperparams=None):
        """
        Load a model from a saved file path.
        
        Args:
            model_path: Path to the saved model
            dataset_name: Name of the dataset the model was trained on
            vocab_size: Vocabulary size
            hyperparams: Hyperparameters to use (will try to load metadata if None)
            
        Returns:
            Loaded model
        """
        # try to load metadata if hyperparams not provided
        if hyperparams is None:
            try:
                # get directory and base filename without extension
                model_dir = os.path.dirname(model_path)
                base_filename = os.path.basename(model_path).rsplit('.', 1)[0]
                framework_prefix = 'lstm' if self.framework == 'lstm' else 'transformer'
                
                # try different metadata file naming patterns
                metadata_patterns = [
                    os.path.join(model_dir, f"{dataset_name}_{framework_prefix}_latest_best_metadata.json"),
                    os.path.join(model_dir, f"{base_filename}_metadata.json")
                ]
                
                metadata_path = None
                for pattern in metadata_patterns:
                    if os.path.exists(pattern):
                        metadata_path = pattern
                        break
                        
                if metadata_path:
                    print(f"Loading model metadata from {metadata_path}")
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        hyperparams = metadata.get('hyperparameters', {})
                else:
                    print(f"No metadata file found. Using default hyperparameters.")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}. Using default hyperparameters.")
        
        # create model with appropriate configuration
        model = self.create_model({"dataset_name": dataset_name, "vocab_size": vocab_size}, hyperparams)
        
        # load model weights
        try:
            if self.framework == 'lstm':
                model.model.load_weights(model_path)
                print("TensorFlow model weights loaded successfully")
            else:  # transformer
                model.load_state_dict(torch.load(model_path))
                print("PyTorch model state dictionary loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
        return model
    
    def train_model(self, model, data_info, hyperparams, epochs=100):
        """
        Train the model.
        
        Args:
            model: Model to train
            data_info: Output from prepare_data
            hyperparams: Model hyperparameters
            epochs: Number of training epochs
        
        Returns:
            Training results
        """
        print(f"{'='*50}")
        print(f"Training model for {epochs} epochs")
        print(f"  - Framework: {self.framework}")
        print(f"  - Dataset: {data_info['dataset_name']}")
        print(f"  - Batch size: {hyperparams.get('batch_size', 32)}")
        print(f"  - Learning rate: {hyperparams.get('learning_rate', 0.001)}")
        print(f"{'='*50}")

        # get data shapes for logging
        if self.framework == 'lstm':
            X_train, Xq_train, Y_train = data_info['train_data']
            X_val, Xq_val, Y_val = data_info['val_data']
            print(f"  - Training data: {len(Y_train)} examples")
            print(f"  - Validation data: {len(Y_val)} examples")
        else:  # transformer
            X_train, masks_train, Y_train = data_info['train_data']
            X_val, masks_val, Y_val = data_info['val_data']
            print(f"  - Training data: {len(Y_train)} examples")
            print(f"  - Validation data: {len(Y_val)} examples")
        
        train_df = data_info['train_df']
        val_df = data_info['val_df']
        
        # create simplified metadata dictionaries
        train_metadata = {
            'dataset_name': data_info['dataset_name']
        }
        
        val_metadata = {
            'dataset_name': data_info['dataset_name']
        }
        
        # add category information if available
        if 'category' in train_df.columns:
            train_metadata['categories'] = train_df['category'].unique().tolist()
            train_metadata['category'] = train_df['category'].value_counts().index[0]
            category_counts = train_df['category'].value_counts().to_dict()
            train_metadata['category_counts'] = category_counts
        
        if 'category' in val_df.columns:
            val_metadata['categories'] = val_df['category'].unique().tolist()
            val_metadata['category'] = val_df['category'].value_counts().index[0]
            val_metadata['category_counts'] = val_df['category'].value_counts().to_dict()
        
        for col in ['n_actors', 'num_nouns']:
            if col in train_df.columns:
                train_metadata[f'avg_{col}'] = train_df[col].mean()
                train_metadata[col] = train_df[col].value_counts().index[0]
        
        for col in ['n_actors', 'num_nouns']:
            if col in val_df.columns:
                val_metadata[f'avg_{col}'] = val_df[col].mean()
                val_metadata[col] = val_df[col].value_counts().index[0]
        
        print("\nDEBUGGING METADATA:")
        print(f"  - Train DataFrame columns: {list(train_df.columns)}")
        print(f"  - Val DataFrame columns: {list(val_df.columns)}")
        
        if 'category' in train_df.columns:
            print(f"  - Train categories: {train_df['category'].unique()}")
        
        print(f"  - Prepared train metadata: {train_metadata}")
        print(f"  - Prepared val metadata: {val_metadata}")
        
        print(f"Starting training loop...")
        results = self.trainer.train(
            model=model,
            train_data=data_info['train_data'],
            val_data=data_info['val_data'],
            metadata_train=train_metadata,
            metadata_val=val_metadata,
            hyperparams=hyperparams,
            epochs=epochs
        )
        
        print(f"\n{'='*50}")
        print(f"Training results:")
        print(f"  - Best validation accuracy: {results['best_val_accuracy']:.4f}")
        print(f"  - Best epoch: {results['best_epoch']}")
        print(f"{'='*50}")

        with tqdm(total=1, desc="Saving best model") as pbar:
            best_model_path = self.save_best_model(data_info, results, hyperparams, is_optimization=False)
            pbar.update(1)
        
        best_model_path = self.save_best_model(data_info, results, hyperparams, is_optimization=False)
        if best_model_path:
            print(f"Saved best model to: {best_model_path}")
        
        return results

    def save_best_model(self, data_info, results, hyperparams, is_optimization=False):
        """
        Save the best model from training to a standard path with clear naming.
        
        Args:
            data_info: Output from prepare_data
            results: Results from training
            hyperparams: The hyperparameters used for this model
            is_optimization: Whether this is from hyperparameter optimization
            
        Returns:
            Path to the saved best model
        """
        dataset_name = data_info.get('dataset_name', 'default')
        best_epoch = results['best_epoch']
        run_id = self.experiment_tracker.run['sys/id'].fetch()
        
        # determine file extension and prefix based on framework
        extension = 'weights.h5' if self.framework == 'lstm' else 'pt'
        framework_prefix = 'lstm' if self.framework == 'lstm' else 'transformer'
        
        # create appropriate subfolder
        subfolder = "optimization" if is_optimization else "training"
        save_dir = os.path.join(self.model_save_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        
        # format key hyperparameters for filename
        hyperparam_str = ""
        key_hyperparams = []
        
        if self.framework == 'lstm':
            if 'hidden_layers' in hyperparams:
                key_hyperparams.append(f"hl{hyperparams['hidden_layers']}")
            if 'dropout' in hyperparams:
                key_hyperparams.append(f"do{hyperparams['dropout']:.2f}")
        else:
            if 'd_model' in hyperparams:
                key_hyperparams.append(f"dm{hyperparams['d_model']}")
            if 'nlayers' in hyperparams:
                key_hyperparams.append(f"nl{hyperparams['nlayers']}")
            if 'dropout' in hyperparams:
                key_hyperparams.append(f"do{hyperparams['dropout']:.2f}")
        
        if 'batch_size' in hyperparams:
            key_hyperparams.append(f"bs{hyperparams['batch_size']}")
        if 'learning_rate' in hyperparams:
            key_hyperparams.append(f"lr{hyperparams['learning_rate']:.4f}")
        
        hyperparam_str = "_".join(key_hyperparams)
        
        raw_dir = os.path.join(self.model_save_dir, "raw")
        source_path = os.path.join(
            raw_dir, 
            f"{dataset_name}_{framework_prefix}_model_{run_id}_epoch_{best_epoch:02d}.{extension}"
        )
        
        if not os.path.exists(source_path):
            print(f"Warning: Could not find best model file at {source_path}")
            # try to find any model file with the run ID
            import glob
            pattern = os.path.join(raw_dir, f"*_{framework_prefix}_model_{run_id}_epoch_*.{extension}")
            model_files = glob.glob(pattern)
            
            if model_files:
                model_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                source_path = model_files[-1]  # get the latest epoch
                print(f"Found alternative model file: {source_path}")
            else:
                # try looking in the main model directory
                pattern = os.path.join(self.model_save_dir, f"*_{framework_prefix}_model_{run_id}_epoch_*.{extension}")
                model_files = glob.glob(pattern)
                
                if model_files:
                    model_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                    source_path = model_files[-1]  # get the latest epoch
                    print(f"Found alternative model file in main directory: {source_path}")
                else:
                    print(f"No model files found matching pattern: {pattern}")
                    return None
        
        # create descriptive target path with hyperparameter info and epoch
        accuracy_str = f"{results['best_val_accuracy']:.4f}".replace(".", "p")
        target_filename = f"{dataset_name}_{framework_prefix}_ep{best_epoch:02d}_{hyperparam_str}_acc{accuracy_str}.{extension}"
        target_path = os.path.join(save_dir, target_filename)
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # copy the model
        shutil.copy(source_path, target_path)
        print(f"Copied best model to {target_path}")
        
        # also create a "latest_best" file for easier programmatic access
        latest_best_path = os.path.join(save_dir, f"{dataset_name}_{framework_prefix}_latest_best.{extension}")
        
        if os.path.exists(latest_best_path) or os.path.islink(latest_best_path):
            try:
                if os.path.islink(latest_best_path):
                    os.unlink(latest_best_path)
                else:
                    os.remove(latest_best_path)
            except (OSError, IOError) as e:
                print(f"Warning: Could not remove existing file at {latest_best_path}: {str(e)}")
                latest_best_path = os.path.join(save_dir, f"{dataset_name}_{framework_prefix}_latest_best_{run_id}.{extension}")
        
        try:
            os.symlink(target_path, latest_best_path)
            print(f"Created symlink to latest best model at {latest_best_path}")
        except (OSError, AttributeError, IOError) as e:
            try:
                shutil.copy(target_path, latest_best_path)
                print(f"Copied latest best model to {latest_best_path}")
            except (OSError, IOError) as e:
                print(f"Warning: Could not create latest_best reference: {str(e)}")
        
        metadata = {
            "dataset_name": dataset_name,
            "framework": self.framework,
            "hyperparameters": hyperparams,
            "best_epoch": best_epoch,
            "best_val_accuracy": results['best_val_accuracy'],
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_optimization": is_optimization
        }
        
        metadata_path = os.path.join(save_dir, f"{dataset_name}_{framework_prefix}_latest_best_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except (OSError, IOError) as e:
            print(f"Warning: Could not save metadata: {str(e)}")
        
        return target_path


    def prepare_metadata_for_evaluation(df_metadata, original_data_info):
        """Prepare proper metadata for evaluation based on the original data."""
        import pandas as pd
        
        if isinstance(df_metadata, pd.DataFrame) and 'category' in df_metadata.columns:
            if 'num_actors' in df_metadata.columns or 'n_actors' in df_metadata.columns:
                return df_metadata
        
        categories = []
        noun_counts = []
        
        for df_key in ['train_df', 'val_df', 'valb_df', 'test_df']:
            if df_key in original_data_info and isinstance(original_data_info[df_key], pd.DataFrame):
                df = original_data_info[df_key]
                
                if 'category' in df.columns and len(categories) == 0:
                    categories = df['category'].unique().tolist()
                
                for col in ['num_actors', 'n_actors', 'num_nouns']:
                    if col in df.columns and len(noun_counts) == 0:
                        noun_counts = sorted(df[col].unique().tolist())
                        break
                        
                if categories and noun_counts:
                    break
        
        if not categories:
            categories = ['simple']
        if not noun_counts:
            noun_counts = [0]
        
        rows = []
        for category in categories:
            for count in noun_counts:
                rows.append({
                    'category': category,
                    'num_actors': count,
                    'dataset_name': original_data_info.get('dataset_name', 'default')
                })
        
        return pd.DataFrame(rows)
        
    def optimize_hyperparameters(self, data_info, param_space, n_trials=10, epochs=50):
        """Modified optimizer that properly handles metadata"""
        from tqdm.auto import tqdm
        import traceback
        import pandas as pd
        
        # create optimizer
        print(f"\n{'='*50}")
        print(f"Hyperparameter Optimization Configuration:")
        print(f"  - Framework: {self.framework}")
        print(f"  - Dataset: {data_info['dataset_name']}")
        print(f"  - Number of trials: {n_trials}")
        print(f"  - Epochs per trial: {epochs}")
        print(f"{'='*50}\n")
        
        print(f"Parameter space:")
        for param in param_space:
            if param['type'] == 'range':
                print(f"  - {param['name']}: range {param['bounds']}" + 
                    (f" (log scale)" if param.get('log_scale', False) else ""))
            elif param['type'] == 'choice':
                print(f"  - {param['name']}: choices {param['values']}")
            elif param['type'] == 'int':
                print(f"  - {param['name']}: integer range {param['bounds']}")
                    
        optimizer = HyperparameterOptimizer(param_space)
        
        optimization_dir = os.path.join(self.model_save_dir, "optimization")
        os.makedirs(optimization_dir, exist_ok=True)
        
        best_model_filename = f"{data_info['dataset_name']}_{self.framework}_best_optimization.{'weights.h5' if self.framework == 'lstm' else 'pt'}"
        best_model_path = os.path.join(optimization_dir, best_model_filename)
        
        # track best model info
        best_trial_results = None
        best_trial_model = None
        best_trial_run_id = None
        best_val_accuracy = 0.0
        best_trial_params = None
        
        trials_progress = tqdm(range(n_trials), desc="Optimization Trials", unit="trial")
        
        print("\nINFO ABOUT DATAFRAMES BEING USED IN OPTIMIZATION:")
        for df_key in ['train_df', 'val_df']:
            if df_key in data_info and isinstance(data_info[df_key], pd.DataFrame):
                df = data_info[df_key]
                print(f"  - {df_key}: {len(df)} rows with columns {list(df.columns)}")
                
                if 'category' in df.columns:
                    print(f"    * Categories: {sorted(df['category'].unique().tolist())}")
                
                for col in ['num_actors', 'n_actors', 'num_nouns']:
                    if col in df.columns:
                        actor_values = sorted(df[col].dropna().unique().tolist()) 
                        print(f"    * {col} values: {actor_values}")
        
        # run trials
        for i in trials_progress:
            # get next parameters to try
            parameters, trial_index = optimizer.get_next_parameters()
            
            print(f"\n{'='*50}")
            print(f"Starting trial {i+1}/{n_trials}")
            print(f"Parameters:")
            for param_name, param_value in parameters.items():
                print(f"  - {param_name}: {param_value}")
            print(f"{'='*50}\n")
            
            run_name = f"hparam_trial_{i+1}"
            self.experiment_tracker.start_run(run_name, parameters)
            
            print(f"Creating model for trial {i+1}...")
            model = self.create_model(data_info, parameters)
            
            print(f"Training model for trial {i+1} with original DataFrames as metadata...")
            
            trial_data_info = data_info.copy()
            if 'train_df' in trial_data_info:
                trial_data_info['train_df'] = trial_data_info['train_df'].copy()
            if 'val_df' in trial_data_info:
                trial_data_info['val_df'] = trial_data_info['val_df'].copy()
                
            if self.framework == 'lstm':
                train_data = trial_data_info['train_data']
                val_data = trial_data_info['val_data']
            else:  # transformer
                train_data = trial_data_info['train_data']
                val_data = trial_data_info['val_data']
            
            metadata_train = trial_data_info['train_df']
            metadata_val = trial_data_info['val_df']
            
            metadata_train['dataset_name'] = trial_data_info.get('dataset_name', 'default')
            metadata_val['dataset_name'] = trial_data_info.get('dataset_name', 'default')
            
            if self.framework == 'lstm':
                trainer = LSTMTrainer(
                    experiment_tracker=self.experiment_tracker,
                    save_directory=self.model_save_dir
                )
            else:  # transformer
                trainer = TransformerTrainer(
                    experiment_tracker=self.experiment_tracker,
                    save_directory=self.model_save_dir
                )
            
            results = trainer.train(
                model=model,
                train_data=train_data,
                val_data=val_data,
                metadata_train=metadata_train,
                metadata_val=metadata_val,
                hyperparams=parameters,
                epochs=epochs
            )
            
            val_accuracy = results['best_val_accuracy']
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_trial_results = results
                best_trial_model = model
                best_trial_run_id = self.experiment_tracker.run['sys/id'].fetch()
                best_trial_params = {k: v for k, v in parameters.items()}  
                
                print(f"New best model found (accuracy: {val_accuracy:.4f}). Saving to {best_model_path}...")
                
                try:
                    if self.framework == 'lstm':
                        model.model.save_weights(best_model_path)
                    else:
                        torch.save(model.state_dict(), best_model_path)
                    print(f"Successfully saved best model to: {best_model_path}")
                except Exception as e:
                    print(f"Error saving best model: {str(e)}")
                    traceback.print_exc()
                    
                    fallback_path = os.path.join(optimization_dir, f"best_model_{self.framework}_{i+1}.{'weights.h5' if self.framework == 'lstm' else 'pt'}")
                    try:
                        if self.framework == 'lstm':
                            model.model.save_weights(fallback_path)
                        else:
                            torch.save(model.state_dict(), fallback_path)
                        best_model_path = fallback_path
                        print(f"Saved best model to fallback path: {best_model_path}")
                    except Exception as e2:
                        print(f"Error saving to fallback path: {str(e2)}")
                
                trials_progress.set_postfix({
                    'best_acc': f"{best_val_accuracy:.4f}",
                    'best_epoch': f"{results['best_epoch']}",
                    'best_trial': f"{i+1}"
                })
            else:
                trials_progress.set_postfix({
                    'acc': f"{val_accuracy:.4f}",
                    'best_acc': f"{best_val_accuracy:.4f}"
                })
                    
            print(f"Trial {i+1} complete. Validation accuracy: {val_accuracy:.4f}")
            optimizer.complete_trial(trial_index, val_accuracy)
            
            self.experiment_tracker.end_run()
        
        best_parameters, best_values = optimizer.get_best_parameters()
        
        print(f"\n{'='*50}")
        print(f"Hyperparameter optimization complete!")
        print(f"Best parameters from optimizer:")
        for param_name, param_value in best_parameters.items():
            print(f"  - {param_name}: {param_value}")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Best model saved to: {best_model_path}")
        print(f"{'='*50}\n")
        
       
        final_params = best_trial_params if best_trial_params else best_parameters
        
        print(f"Using parameters for final model:")
        for param_name, param_value in final_params.items():
            print(f"  - {param_name}: {param_value}")
        
        final_model = self.create_model(data_info, final_params)
        
        load_successful = False
        try:
            if self.framework == 'lstm':
                final_model.model.load_weights(best_model_path)
            else:
                final_model.load_state_dict(torch.load(best_model_path))
            print(f"Successfully loaded best model from: {best_model_path}")
            load_successful = True
        except Exception as e:
            print(f"Error loading best model: {str(e)}")
            print("Will train a new model with best parameters instead")
        
        final_results = None
        if not load_successful or epochs*2 > 0:
            print(f"Training final model with best parameters (for {epochs*2} epochs)...")
            self.experiment_tracker.start_run("best_model", final_params)
            
            if not load_successful:
                print("Creating fresh model with best parameters...")
                final_model = self.create_model(data_info, final_params)
            
            final_data_info = data_info.copy()
            if 'train_df' in final_data_info:
                final_data_info['train_df'] = final_data_info['train_df'].copy()
            if 'val_df' in final_data_info:
                final_data_info['val_df'] = final_data_info['val_df'].copy()
                
            if self.framework == 'lstm':
                train_data = final_data_info['train_data']
                val_data = final_data_info['val_data']
            else:  # transformer
                train_data = final_data_info['train_data']
                val_data = final_data_info['val_data']
            
            metadata_train = final_data_info['train_df']
            metadata_val = final_data_info['val_df']
            
            metadata_train['dataset_name'] = final_data_info.get('dataset_name', 'default')
            metadata_val['dataset_name'] = final_data_info.get('dataset_name', 'default')
            
            if self.framework == 'lstm':
                trainer = LSTMTrainer(
                    experiment_tracker=self.experiment_tracker,
                    save_directory=self.model_save_dir
                )
            else:  # transformer
                trainer = TransformerTrainer(
                    experiment_tracker=self.experiment_tracker,
                    save_directory=self.model_save_dir
                )
            
            final_results = trainer.train(
                model=final_model,
                train_data=train_data,
                val_data=val_data,
                metadata_train=metadata_train,
                metadata_val=metadata_val,
                hyperparams=final_params,
                epochs=epochs*2
            )
            
            final_model_path = os.path.join(optimization_dir, f"{data_info['dataset_name']}_{self.framework}_final.{'weights.h5' if self.framework == 'lstm' else 'pt'}")
            
            try:
                if self.framework == 'lstm':
                    final_model.model.save_weights(final_model_path)
                else:
                    torch.save(final_model.state_dict(), final_model_path)
                print(f"Saved final model to: {final_model_path}")
            except Exception as e:
                print(f"Error saving final model: {str(e)}")
                    
            self.experiment_tracker.end_run()
        
        return {
            'best_parameters': final_params,
            'best_model_path': best_model_path,
            'final_model': final_model,
            'final_results': final_results,
            'best_accuracy': best_val_accuracy,
            'train_df': data_info['train_df'],  
            'val_df': data_info['val_df']
        }
            
    def evaluate_model(self, model, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            test_data: Test data tuple
        
        Returns:
            Evaluation metrics
        """
        print(f"Starting model evaluation...")
        
        if self.framework == 'lstm':
            X_test, Xq_test, Y_test = test_data
            print(f"  - Test data: {len(Y_test)} examples")
            print(f"  - Test data shapes: X={X_test.shape}, Xq={Xq_test.shape}, Y={Y_test.shape}")
            
            print(f"Evaluating LSTM model...")
            loss, accuracy = model.model.evaluate([X_test, Xq_test], Y_test)
            
            # make predictions
            print(f"Generating predictions...")
            predictions = model.model.predict([X_test, Xq_test])
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_true = Y_test
            
            print(f"Evaluation complete:")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - Loss: {loss:.4f}")
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': y_pred,
                'true_values': y_true
            }
        
        else:  # transformer
            X_test, masks_test, Y_test = test_data
            print(f"  - Test data: {len(Y_test)} examples")
            print(f"  - Test data shapes: X={X_test.shape}, masks={masks_test.shape}, Y={Y_test.shape}")
            
            model.eval()
            
            criterion = torch.nn.BCEWithLogitsLoss()
            
            print(f"Evaluating transformer model...")
            with torch.no_grad():
                outputs = model(X_test, masks_test)
            
                # ensure outputs and labels have compatible shapes
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, Y_test).item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                accuracy = (predictions == Y_test).float().mean().item()
            
                # convert to numpy for consistency
                y_pred = predictions.cpu().numpy()
                y_true = Y_test.cpu().numpy()

            print(f"Evaluation complete:")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - Loss: {loss:.4f}")

        if hasattr(self, 'data_info') and 'test_df' in self.data_info:
            test_df = self.data_info['test_df']
            trace_evaluation_data_flow({"test_df": test_df}, {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': len(y_pred)  
            }, "After Model Evaluation")
        else:
            trace_evaluation_data_flow({}, {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': len(y_pred) 
            }, "After Model Evaluation")   

            return {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': y_pred,
                'true_values': y_true
            }