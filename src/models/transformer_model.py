# pytorch transformer model for binary question answering

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 150):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer model for binary question answering."""
    
    def __init__(self, ntoken: int, d_model: int = 256, nhead: int = 2, 
                 d_hid: int = 569, nlayers: int = 4, dropout: float = 0.413):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=500)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.classifier = nn.Linear(d_model, 1)
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.3
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
    
    def forward(self, src, src_mask):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        cls_output = output[:, 0, :]  # use the first token ([CLS]) for classification
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)