"""
Neural Network Models for Scalping Trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttentionModel(nn.Module):
    """Multi-Head Attention Model for Scalping"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, num_classes=3):
        super(MultiHeadAttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Project to hidden size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Attention pooling
        attn_output, _ = self.attention_pool(x, x, x)
        x = self.layer_norm(x + attn_output)
        
        # Global pooling
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CNNLSTMModel(nn.Module):
    """CNN-LSTM Hybrid Model"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes=3):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layers
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Permute for CNN (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Permute back for LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        out = self.fc(context_vector)
        
        return out

def create_model(model_config, input_size):
    """Factory function to create model based on configuration"""
    
    architecture = model_config['model']['architecture']
    hidden_size = model_config['model']['hidden_size']
    num_layers = model_config['model']['num_layers']
    dropout = model_config['model']['dropout']
    
    if architecture == 'multi_head_attention':
        num_heads = model_config['model']['num_heads']
        return MultiHeadAttentionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    elif architecture == 'cnn_lstm':
        return CNNLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")