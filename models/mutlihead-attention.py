import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAUROC
from torch.optim import Adam
import tqdm
import os
import numpy

BATCH_SIZE = 40
LEARNING_RATE = 0.001
EPOCHS = 20
LSTM_NODES = 256
NUM_SENTENCES = 40
SENTENCE_LENGTH = 98
VOCAB_SIZE = 49152
EMBEDDING_SIZE = 4096
NUM_EPOCHS = 20

### as of 7-22-25 this archiecture has not been debugged and is not functional. It is a work in progress. ###

cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device('cuda')

# ... (previous imports remain the same)

# Add these new hyperparameters
NUM_HEADS = 8             # For multi-head attention
FF_DIM = 512              # Feed-forward dimension
LOSS_ALPHA = 0.75         # Weight for positive class in loss function

# ... (previous setup code remains the same until model definition)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        return self.out_proj(attn_output)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, batch_first, bidirectional, dropout, 
                 pretrained_weights, batch_size, sentence_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.rnn = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout, 
            batch_first=batch_first
        ).to(device)
        
        # Multi-head attention setup
        self.attention = MultiHeadAttention(
            embed_dim=hidden_dim * 2, 
            num_heads=NUM_HEADS, 
            dropout=dropout
        ).to(device)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim * 2).to(device)
        self.ln2 = nn.LayerNorm(hidden_dim * 2).to(device)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, FF_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(FF_DIM, hidden_dim * 2)
        ).to(device)
        
        # Output layers
        self.fc = nn.Linear(hidden_dim * 2, output_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.embedding.weight.data.copy_(pretrained_weights).to(device)

    def forward(self, text):
        text = text.to(device)
        batch_size, sentence_length, num_sentences = text.size()
        text = text.view(batch_size, sentence_length * num_sentences)
        
        # Embedding layer
        embedded = self.dropout(self.embedding(text))
        
        # LSTM layer
        lstm_output, (hidden, _) = self.rnn(embedded)
        
        # Multi-head attention with residual connection
        attn_output = self.attention(lstm_output)
        attn_output = self.ln1(lstm_output + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(attn_output)
        ffn_output = self.ln2(attn_output + ffn_output)
        
        # Attention pooling
        attention_weights = torch.softmax(self.fc_attn(ffn_output), dim=1)
        context_vector = torch.sum(ffn_output * attention_weights, dim=1)
        
        # Final output
        output = self.fc_out(self.dropout(context_vector))
        return torch.sigmoid(output)

# ... (rest of the setup remains the same)

# Modify loss function to handle class imbalance
class WeightedBCELoss(nn.Module):
    def __init__(self, alpha=LOSS_ALPHA):
        super().__init__()
        self.alpha = alpha  # Weight for positive class
        
    def forward(self, inputs, targets):
        loss = - (self.alpha * targets * torch.log(inputs + 1e-7) - 
                 (1 - self.alpha) * (1 - targets) * torch.log(1 - inputs + 1e-7))
        return torch.mean(loss)

# Update model initialization
model = LSTMClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_SIZE,  
    hidden_dim=LSTM_NODES,
    output_dim=40,  
    n_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.3,  # Slightly reduced due to layer norm
    pretrained_weights=word_vectors,
    batch_size=BATCH_SIZE,
    sentence_length=SENTENCE_LENGTH
)

# Use the new weighted loss
criterion = WeightedBCELoss().to(device)

# ... (training loop remains the same)