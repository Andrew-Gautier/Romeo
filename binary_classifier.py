"""
LSTM Classifier Model for Vulnerability Detection
Separated from training script for modularity and reuse.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with multi-head attention mechanism for binary vulnerability classification.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of token embeddings
        hidden_dim (int): Dimension of LSTM hidden state
        output_dim (int): Output dimension (1 for binary classification)
        n_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        dropout (float): Dropout probability
        n_heads (int): Number of attention heads
        pretrained_weights (torch.Tensor, optional): Pretrained embedding weights
        device (torch.device): Device to place the model on
    """
    
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        output_dim=1, 
        n_layers=2, 
        bidirectional=True, 
        dropout=0.5,
        n_heads=8,
        pretrained_weights=None,
        device=None
    ):
        super().__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_heads = n_heads
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
        
        # LSTM layer
        self.rnn = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout if n_layers > 1 else 0, 
            batch_first=True
        )
        
        # Multi-head attention layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        # NOTE: Don't call self.to(self.device) here - let the training script handle
        # device placement. This allows DataParallel to work correctly.

    def forward(self, text):
        """
        Forward pass through the model.
        
        Args:
            text (torch.Tensor): Input tensor of shape [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Output probabilities of shape [batch_size, 1]
        """
        # NOTE: Don't move text to self.device here - DataParallel already handles device placement
        # The input tensor is already on the correct GPU for each replica
        
        # Ensure text is 2D: [batch_size, seq_length]
        if text.dim() == 3:
            batch_size, seq_length, _ = text.size()
            text = text.view(batch_size, -1)
        
        # Embedding
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch_size, seq_length, embedding_dim]
        
        # LSTM
        lstm_output, (hidden, cell) = self.rnn(embedded)
        # lstm_output = [batch_size, seq_length, hidden_dim * num_directions]
        
        # Multi-head self-attention
        # Query, Key, Value are all the LSTM output (self-attention)
        attn_output, attn_weights = self.multihead_attention(
            lstm_output, lstm_output, lstm_output
        )
        # attn_output = [batch_size, seq_length, hidden_dim * num_directions]
        
        # Residual connection + layer normalization
        attn_output = self.layer_norm(lstm_output + attn_output)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(attn_output, dim=1)
        # pooled_output = [batch_size, hidden_dim * num_directions]
        
        # Output layer
        output = self.fc(self.dropout(pooled_output))
        # output = [batch_size, 1]
        
        # Sigmoid for binary classification
        output = torch.sigmoid(output)
        
        return output
    
    def get_config(self):
        """Return model configuration as a dictionary."""
        return {
            'vocab_size': self.embedding.num_embeddings,
            'embedding_dim': self.embedding.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.fc.out_features,
            'n_layers': self.n_layers,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout.p,
            'n_heads': self.n_heads,
        }


def create_model(config, pretrained_weights=None, device=None):
    """
    Factory function to create an LSTMClassifier from a config dict.
    
    Args:
        config (dict): Model configuration
        pretrained_weights (torch.Tensor, optional): Pretrained embedding weights
        device (torch.device, optional): Device to place the model on
        
    Returns:
        LSTMClassifier: Initialized model on the specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config.get('output_dim', 1),
        n_layers=config.get('n_layers', 2),
        bidirectional=config.get('bidirectional', True),
        dropout=config.get('dropout', 0.5),
        n_heads=config.get('n_heads', 8),
        pretrained_weights=pretrained_weights,
        device=device
    )
    
    # Move model to device (important for DataParallel compatibility)
    return model.to(device)
