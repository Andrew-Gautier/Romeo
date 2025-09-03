import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAUROC
from torch.optim import Adam
import tqdm
import os
import math

BATCH_SIZE = 40
LEARNING_RATE = 0.001
EPOCHS = 20
ATTENTION_HEADS = 8
ATTENTION_DIM = 256
NUM_SENTENCES = 50
SENTENCE_LENGTH = 98
VOCAB_SIZE = 49152
EMBEDDING_SIZE = 4096
NUM_EPOCHS = 20

# Make sure tensors are all on GPU
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device('cuda')

train_sequences_tensor = torch.load("cwe_train_sequences.pt").long()
train_labels = torch.load("cwe_train_labels.pt") 
train_dataset = TensorDataset(train_sequences_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

val_sequences_tensor = torch.load('cwe_val_sequences.pt').long()
val_labels = torch.load('cwe_val_labels.pt') 
val_dataset = TensorDataset(val_sequences_tensor, val_labels)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = False)

test_sequences_tensor = torch.load("cwe_test_sequences.pt").long()
test_labels = torch.load("cwe_test_labels.pt") 
test_dataset = TensorDataset(test_sequences_tensor, test_labels)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = False)

torch.manual_seed(691)

# TODO: Model checkpoints, Model saving
try:
    pretrained_weights = torch.load('aix3-7b-base (1).pt')
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Failed to load weights: {e}")

# Once you've identified the key for the embeddings, you can extract them like this:
word_vectors = pretrained_weights['tok_embeddings.weight']
print(word_vectors.shape)

def save_checkpoint(state, epoch, checkpoint_path="/attention_check_point_path"):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    filename = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
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
        
    def forward(self, value, key, query, mask):
        attention, attention_weights = self.attention(value, key, query, mask)
        
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention_weights

class SelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, dropout, pretrained_weights):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.position_encoding = self.create_positional_encoding(embedding_dim, 5000).to(device)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size=embedding_dim,
                heads=num_heads,
                dropout=dropout,
                forward_expansion=4
            ) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embedding_dim, output_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.embedding.weight.data.copy_(pretrained_weights).to(device)
        
    def create_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension
        
    def forward(self, text):
        text = text.to(device)
        batch_size, seq_len = text.shape
        
        # Embedding + positional encoding
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        embedded += self.position_encoding[:, :seq_len, :]
        embedded = self.dropout(embedded)
        
        # Self-attention (no mask for classification tasks)
        mask = None
        x = embedded
        attention_weights = []
        
        for layer in self.layers:
            x, weights = layer(x, x, x, mask)
            attention_weights.append(weights)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # [batch_size, embedding_dim]
        
        # Final classification layer
        output = self.fc_out(x)
        output = torch.sigmoid(output)
        
        return output

model = SelfAttentionClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_SIZE,
    hidden_dim=ATTENTION_DIM,
    output_dim=50,
    num_heads=ATTENTION_HEADS,
    num_layers=2,
    dropout=0.1,
    pretrained_weights=word_vectors
)
print(model)

model.eval()

with torch.no_grad():
    # Get the first batch of data from the training loader
    batch_sequences, batch_labels = next(iter(train_loader))
        
    # Pass the batch of sequences through the model
    outputs = model(batch_sequences).to(device)
    
def train(model, iterator, optimizer, criterion, epoch, device, checkpoint_path="attention_checkpoints"):
    epoch_loss = 0
    model.train()
    
    for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Training'):
        # Move data to the device
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_sequences)
        
        predictions = predictions.view(-1, 50).float()  # Flatten if necessary
        batch_labels = batch_labels.view(-1, 50).float()  # Ensure labels are correctly shaped
        
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    # Save checkpoint after the epoch
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': epoch_loss / len(iterator),
    }, epoch, checkpoint_path=checkpoint_path)    
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    model.eval()
    auroc = BinaryAUROC().to(device)  # Initialize AUROC metric

    with torch.no_grad():
        for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Evaluation'):
            # Move data to the device
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            predictions = model(batch_sequences)
            predictions = predictions.view(-1, 50).float()  # Flatten if necessary
            batch_labels = batch_labels.view(-1, 50).float()  # Ensure labels are correctly shaped
            
            probabilities = torch.sigmoid(predictions)  # Convert logits to probabilities
            
            # Update AUROC computation
            auroc.update(probabilities, batch_labels.int())
            
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
    
    auroc_score = auroc.compute()  # Compute the final AUROC score
    auroc.reset()  # Reset AUROC metric for future use
    
    return epoch_loss / len(iterator), auroc_score.item()

# MAIN TRAINING LOOP

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss().to(device)

# Define the number of epochs
N_EPOCHS = 40
# Implement a basic early stopping counter
best_valid_loss = float('inf')
epochs_since_improvement = 0

# Store the loss values for plotting
train_losses = []
valid_losses = []
valid_aurocs = []

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, epoch, device)
    valid_loss, valid_auroc = evaluate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_aurocs.append(valid_auroc)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, Val. AUROC: {valid_auroc:.3f}')
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_since_improvement = 0  # Reset counter
    else:
        epochs_since_improvement += 1  # Increment counter
    
    # Stop training if validation loss hasn't improved for 3 consecutive epochs
    if epochs_since_improvement == 3:
        print("Stopping early due to no improvement in validation loss for 3 consecutive epochs.")
        break

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('attention_loss_plot.png')
plt.close()

# Optionally, plot the validation AUROC scores
plt.figure(figsize=(10, 5))
plt.plot(valid_aurocs, label='Validation AUROC')
plt.xlabel('Epochs')
plt.ylabel('AUROC')
plt.legend()
plt.ylim(0.5, 1.0)  # Set the y-axis to scale between 0.5 and 1
plt.xlim(0, 20)  # Set the x-axis to scale between 0 and 20
plt.savefig('attention_auroc_plot.png')
plt.close()

# Evaluate the model on the test dataset
print("\n--- Test Set Evaluation ---")
test_loss, test_auroc = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.3f}, Test AUROC: {test_auroc:.3f}')

# Save the final model
final_model_path = 'self_attention_final_model.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'test_loss': test_loss,
    'test_auroc': test_auroc,
    'config': {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_SIZE,
        'hidden_dim': ATTENTION_DIM,
        'output_dim': 50,
        'num_heads': ATTENTION_HEADS,
        'num_layers': 2,
        'dropout': 0.1,
    }
}, final_model_path)
print(f'Model saved to {final_model_path}')