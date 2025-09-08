import torch
import torch.nn as nn
import math
device = torch.device('cuda')

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        ### Project to Q,K,V at once ###
        self.qkv_proj = nn.Linear(embed_size, 3* embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        # Project input to Q, K, V
        qkv = self.qkv_proj(x)
        # Reshape to [N, seq_len, heads, 3 * head_dim]
        qkv = qkv.reshape(N, seq_length, self.heads, 3 * self.head_dim)
        # Split into Q, K, V tensors of shape [N, seq_len, heads, head_dim]
        queries, keys, values = torch.chunk(qkv, 3, dim=-1)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (0.5)), dim=-1) # Use head_dim, not embed_size

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        # Re-assemble all head outputs side-by-side
        out = out.reshape(N, seq_length, self.embed_size)
        out = self.fc_out(out)
        return out, attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask): 
        attention, attention_weights = self.attention(x, mask) 
        x = self.dropout(self.norm1(attention + x)) # Added skip connection to input x
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention_weights

class SelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, dropout, pretrained_weights):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = self.create_positional_encoding(embedding_dim, 5000)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size=embedding_dim,
                heads=num_heads,
                dropout=dropout,
                forward_expansion=4
            ) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding.weight.data.copy_(pretrained_weights)
        
    def create_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension
        
    def forward(self, text):
        # Handle input reshaping first
        original_shape = text.shape
        if len(original_shape) == 3:
            batch_size, num_sentences, seq_len = original_shape
            text = text.view(-1, seq_len)  # Flatten to [batch_size * num_sentences, seq_len]
            needs_reshaping = True
        else:
            batch_size, seq_len = original_shape
            num_sentences = None
            needs_reshaping = False

        # Embedding + positional encoding (now text is always 2D [N, seq_len])
        embedded = self.embedding(text)  # [N, seq_len, embedding_dim]
        embedded = embedded + self.position_encoding[:, :seq_len, :]
        embedded = self.dropout(embedded)

        # Self-attention
        mask = None
        x = embedded
        attention_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            attention_weights.append(weights)

        # Global average pooling
        x = torch.mean(x, dim=1)  # [N, embedding_dim]

        # Reshape back if necessary
        if needs_reshaping:
            # x is now [batch_size * num_sentences, embedding_dim]
            x = x.view(batch_size, num_sentences, -1)
            # Average across sentences
            x = torch.mean(x, dim=1)  # [batch_size, embedding_dim]
        # Else, x is already [batch_size, embedding_dim]

        output = self.fc_out(x)
        output = torch.sigmoid(output)
        return output
