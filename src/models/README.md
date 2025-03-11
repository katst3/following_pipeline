# Models Directory

This directory contains model implementations for binary QA in both LSTM (TensorFlow) and transformer (PyTorch) frameworks. The models are designed to process preprocessed story data and handle binary question-answering directional tasks.

## Contents

- **lstm_model.py**: TensorFlow-based LSTM model implementation
- **transformer_model.py**: PyTorch-based Transformer model implementation

## Model Architectures

### LSTM Model (TensorFlow)

The `QA_Model` class implements an LSTM-based architecture for story question answering:

#### Architecture Overview:
1. **Dual Encoder Design**:
   - Separate encoders for story and question inputs
   - Each encoder consists of an Embedding layer followed by an LSTM layer

2. **Input Processing**:
   - Story input: Preprocessed and padded sequence of word indices
   - Question input: Preprocessed and padded sequence of word indices

3. **Network Structure**:
   - Embedding layers (64-dimensional) for both story and question inputs
   - LSTM layers to encode sequential information
   - Concatenation of story and question encodings
   - Dense layer with ReLU activation
   - Dropout for regularization
   - Final sigmoid output layer for binary classification

4. **Regularization**:
   - Dropout in LSTM and dense layers
   - L1 and L2 regularization in the final dense layer

#### Hyperparameters:
- `hidden_layers`: Size of hidden representations (default: 74)
- `dropout_rate`: Dropout probability (default: 0.39)
- `l1_regul`: L1 regularization strength (default: 0.00005)
- `l2_regul`: L2 regularization strength (default: 0.00003)
- `batch_size`: Number of samples per gradient update (default: 128)
- `learning_rate`: Learning rate for Adam optimizer (default: 0.001)

### Transformer Model (PyTorch)

The `TransformerModel` class implements a Transformer-based architecture:

#### Architecture Overview:
1. **Components**:
   - Embedding layer for token representation
   - Positional encoding to maintain sequence order information
   - Transformer encoder layers with multi-head self-attention
   - Linear classifier layer for binary prediction

2. **Input Processing**:
   - Combined story and question sequence with special tokens
   - Attention mask to handle padding and focus on relevant tokens

3. **Network Structure**:
   - Token embeddings scaled by sqrt(d_model)
   - Positional encoding added to embeddings
   - Multiple Transformer encoder layers
   - Classification based on the [CLS] token representation
   - Sigmoid activation (implicit in loss function) for binary output

4. **Positional Encoding**:
   - Sine and cosine functions of different frequencies
   - Allows the model to understand token positions in the sequence

#### Hyperparameters:
- `d_model`: Embedding dimension (default: 256)
- `nhead`: Number of attention heads (default: 2)
- `d_hid`: Dimension of feedforward network (default: 569)
- `nlayers`: Number of transformer layers (default: 4)
- `dropout`: Dropout probability (default: 0.413)

## Integration with Training Components

The models are designed to work with their respective trainers:

1. **LSTM Model**:
   - Works with `LSTMTrainer` and `LSTMNeptuneLogger`
   - Outputs are processed by the trainer to compute metrics
   - Uses Keras' Model API for built-in training loop and callbacks

2. **Transformer Model**:
   - Works with `TransformerTrainer` and `TransformerNeptuneLogger`
   - Forward pass is called explicitly in the training loop
   - Outputs are processed manually to compute metrics