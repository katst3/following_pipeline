# tensorflow LSTM model for binary question answering

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Embedding, Dropout, Dense, concatenate
from tensorflow.keras.optimizers import Adam

class QA_Model:
    """LSTM model for story question answering."""
    
    def __init__(self, vocab_size, max_story_len, max_question_len, hyperparams):
        self.vocab_size = vocab_size + 2  
        self.max_story_len = max_story_len
        self.max_question_len = max_question_len
        self.hyperparams = hyperparams
        self.model = self.build_model()
    
    def build_model(self):
        # get hyperparameters with defaults (all for 2dir)
        hidden_layers = self.hyperparams.get("hidden_layers", 74)  
        dropout_rate = self.hyperparams.get("dropout", 0.39)       
        l1_regul = self.hyperparams.get("l1_regul", 0.00005)       
        l2_regul = self.hyperparams.get("l2_regul", 0.00003)       
        batch_size = self.hyperparams.get("batch_size", 128)       
        learning_rate = self.hyperparams.get("learning_rate", 0.001)  

        input_sequence = Input((self.max_story_len,))
        question = Input((self.max_question_len,))

        # encoder for input_sequence
        input_encoder = Sequential()
        input_encoder.add(Embedding(input_dim=self.vocab_size, output_dim=64))
        input_encoder.add(LSTM(hidden_layers, dropout=dropout_rate))
        
        # encoder for question
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=self.vocab_size, output_dim=64, input_length=self.max_question_len))
        question_encoder.add(LSTM(hidden_layers, dropout=dropout_rate))
        
        input_encoded = input_encoder(input_sequence)
        question_encoded = question_encoder(question)
        
        # combine encoded inputs
        combined = concatenate([input_encoded, question_encoded])
        
        # final dense layer for prediction
        answer = Dense(hidden_layers, activation='relu')(combined)
        answer = Dropout(dropout_rate)(answer)
        answer = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_regul, l2=l2_regul))(answer)
        
        model = Model(inputs=[input_sequence, question], outputs=answer)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def load_weights(self, filepath):
        """Load model weights from a file."""
        self.model.load_weights(filepath)
        
    def evaluate(self, inputs_test, queries_test, answers_test):
        """Evaluate the model on test data."""
        return self.model.evaluate([inputs_test, queries_test], answers_test)

    def predict(self, inputs_test, queries_test):
        """Generate predictions for the given inputs."""
        return self.model.predict([inputs_test, queries_test])