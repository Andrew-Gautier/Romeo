import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAUROC
import tqdm
import os
import uuid
from datetime import datetime
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from transformer_model import SelfAttentionClassifier

NUM_SENTENCES = 60
DEFAULT_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 20,
    "attention_heads": 8,
    "attention_dim": 256,
    "num_sentences": 60,
    "sentence_length": 64,
    "vocab_size": 49152,
    "embedding_size": 4096,
    "dropout": 0.1,
    "num_layers": 2
}

# Make sure tensors are all on GPU
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device('cuda' if cuda_available else 'cpu')

def generate_unique_id():
    """Generate a unique identifier for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"

def load_data(config):
    """Load data and create data loaders based on config"""
    batch_size = config["batch_size"]
    
    # Load training data
    train_sequences_tensor = torch.load("tensors_run0/cwe_train_sequences.pt").long().to(device)
    train_labels = torch.load("tensors_run0/cwe_train_labels.pt").to(device)
    train_dataset = TensorDataset(train_sequences_tensor, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)
    
    # Load validation data
    val_sequences_tensor = torch.load('tensors_run0/cwe_val_sequences.pt').long().to(device)
    val_labels = torch.load('tensors_run0/cwe_val_labels.pt').to(device)
    val_dataset = TensorDataset(val_sequences_tensor, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False)

    # Load test data
    test_sequences_tensor = torch.load("tensors_run0/cwe_test_sequences.pt").long().to(device)
    test_labels = torch.load("tensors_run0/cwe_test_labels.pt").to(device)
    test_dataset = TensorDataset(test_sequences_tensor, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False)
    
    return train_loader, val_loader, test_loader

def load_word_vectors():
    """Load pretrained word vectors"""
    try:
        pretrained_weights = torch.load('aix3-7b-base (1).pt', map_location=device)
        print("Weights loaded successfully.")
        # Extract the word vectors
        word_vectors = pretrained_weights['tok_embeddings.weight']
        print(f"Word vectors shape: {word_vectors.shape}")
        return word_vectors
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return None

def save_checkpoint(model, optimizer, epoch, loss, auroc, config, run_id, checkpoint_path="checkpoints"):
    """Save a checkpoint with a unique identifier"""
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    filename = os.path.join(checkpoint_path, f"checkpoint_run_{run_id}_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'auroc': auroc,
        'config': config
    }, filename)
    print(f"Checkpoint saved to {filename}")
    
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    epoch_loss = 0
    model.train()
    
    for batch_sequences, batch_labels in tqdm.tqdm(dataloader, desc='Training'):
        # Move data to the device
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_sequences)

        predictions = predictions.view(-1, NUM_SENTENCES).float()  # Flatten if necessary
        batch_labels = batch_labels.view(-1, NUM_SENTENCES).float()  # Ensure labels are correctly shaped

        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model"""
    epoch_loss = 0
    model.eval()
    auroc = BinaryAUROC().to(device)  # Initialize AUROC metric

    with torch.no_grad():
        for batch_sequences, batch_labels in tqdm.tqdm(dataloader, desc='Evaluation'):
            # Move data to the device
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            predictions = model(batch_sequences)
            predictions = predictions.view(-1, NUM_SENTENCES).float()  # Flatten if necessary
            batch_labels = batch_labels.view(-1, NUM_SENTENCES).float()  # Ensure labels are correctly shaped

            # Update AUROC computation
            auroc.update(predictions, batch_labels.int())
            
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
    
    auroc_score = auroc.compute()  # Compute the final AUROC score
    auroc.reset()  # Reset AUROC metric for future use
    
    return epoch_loss / len(dataloader), auroc_score.item()
    
def train_model(config, checkpoint_dir=None, report_to_tune=True):
    # Set seed for reproducibility
    torch.manual_seed(config.get("seed", 691))
    torch.cuda.manual_seed_all(config.get("seed", 691))
    # Generate a unique run ID
    run_id = generate_unique_id()
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config)
    
    # Load word vectors
    word_vectors = load_word_vectors()
    if word_vectors is None:
        raise ValueError("Failed to load word vectors")
    
    # Create model
    model = SelfAttentionClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_size"],
        hidden_dim=config["attention_dim"],
        output_dim=60,  # Fixed output dimension
        num_heads=config["attention_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pretrained_weights=word_vectors
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCELoss().to(device)
    
    # Load checkpoint if provided
    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Training loop
    best_valid_loss = float('inf')
    epochs_since_improvement = 0
    train_losses = []
    valid_losses = []
    valid_aurocs = []
    
    for epoch in range(config["epochs"]):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluate on validation set
        valid_loss, valid_auroc = evaluate_model(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_aurocs.append(valid_auroc)
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, Val. AUROC: {valid_auroc:.3f}')
        
        # Report metrics to Ray Tune if requested
        if report_to_tune:
            tune.report(
                train_loss=train_loss,
                val_loss=valid_loss,
                val_auroc=valid_auroc,
                epoch=epoch
            )
        
        # Save checkpoint with unique identifier
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=valid_loss,
            auroc=valid_auroc,
            config=config,
            run_id=run_id
        )
        
        # Early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        if epochs_since_improvement == 3:
            print("Stopping early due to no improvement in validation loss for 3 consecutive epochs.")
            break
    
    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    test_loss, test_auroc = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.3f}, Test AUROC: {test_auroc:.3f}')
    
    # Save the final model with a unique name
    final_model_path = f'models/self_attention_final_model_{run_id}.pt'
    # Ensure the directory exists
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_loss': test_loss,
        'test_auroc': test_auroc,
        'config': config
    }, final_model_path)
    print(f'Model saved to {final_model_path}')
    
    # Plot and save training curves
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_dir}/attention_loss_plot_{run_id}.png')
    plt.close()
    
    # Plot the validation AUROC scores
    plt.figure(figsize=(10, 5))
    plt.plot(valid_aurocs, label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.legend()
    plt.ylim(0.5, 1.0)  # Set the y-axis to scale between 0.5 and 1
    plt.xlim(0, len(valid_aurocs))  # Set the x-axis to scale appropriately
    plt.savefig(f'{plot_dir}/attention_auroc_plot_{run_id}.png')
    plt.close()
    
    return test_auroc

def run_hyperparameter_tuning():
    """Run hyperparameter tuning using Ray Tune"""
    # Define search space for hyperparameters
    search_space = {
        "batch_size": tune.choice([16, 32, 40, 64]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "attention_heads": tune.choice([4, 8, 16]),
        "attention_dim": tune.choice([128, 256, 512]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout": tune.choice([0.0, 0.1, 0.3, 0.5]),
        # Fixed parameters
        "num_sentences": 60,
        "sentence_length": 64,
        "vocab_size": 49152,
        "embedding_size": 4096,
        "epochs": 20,
        "seed": 691
    }
    
    # Set up scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_auroc",
        mode="max",
        max_t=20,  # Maximum number of epochs
        grace_period=5,  # Minimum number of epochs before stopping
        reduction_factor=2
    )
    
    # Configure Ray Tune resources
    resources_per_trial = {"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0}
    
    # Run hyperparameter tuning
    result = tune.run(
        tune.with_parameters(train_model),
        resources_per_trial=resources_per_trial,
        config=search_space,
        num_samples=10,  # Number of hyperparameter combinations to try
        scheduler=scheduler,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        name="self_attention_tuning"
    )
    
    # Get the best trial
    best_trial = result.get_best_trial("val_auroc", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation AUROC: {best_trial.last_result['val_auroc']}")
    
    # Train a model with the best hyperparameters
    best_config = best_trial.config
    print("Training model with best hyperparameters...")
    train_model(best_config, report_to_tune=False)

def run_single_model(config=None):
    """Run a single model with default or specified configuration"""
    if config is None:
        config = DEFAULT_CONFIG
    print(f"Running model with configuration: {config}")
    return train_model(config, report_to_tune=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train self-attention models for code vulnerability detection')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    
    args = parser.parse_args()
    
    if args.tune:
        print("Running hyperparameter tuning...")
        run_hyperparameter_tuning()
    else:
        config = DEFAULT_CONFIG
        if args.config:
            import json
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        run_single_model(config)    