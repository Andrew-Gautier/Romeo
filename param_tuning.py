


### as of 7-22-25 this archiecture has not been debugged and is not functional. It is a work in progress. ###

# Add these new imports at the top
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Hyperparameter search space configuration
config = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([32, 64, 128]),
    "lstm_nodes": tune.choice([128, 256, 512]),
    "dropout": tune.uniform(0.3, 0.6),
    "weight_decay": tune.loguniform(1e-6, 1e-3)
}

# Modified training function for hyperparameter tuning
def train_with_tune(config, checkpoint_dir=None):
    # Initialize model with tuned parameters
    model = LSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_SIZE,
        hidden_dim=config["lstm_nodes"],
        output_dim=40,
        n_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=config["dropout"],
        pretrained_weights=word_vectors,
        batch_size=config["batch_size"],
        sentence_length=SENTENCE_LENGTH
    ).to(device)
    
    # Create new data loaders with tuned batch size
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    optimizer = Adam(model.parameters(), 
                   lr=config["lr"],
                   weight_decay=config["weight_decay"])
    criterion = nn.BCELoss().to(device)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, epoch, device)
        valid_loss, valid_auroc = evaluate(model, test_loader, criterion, device)
        
        # Report metrics to Tune
        tune.report(
            loss=valid_loss,
            auroc=valid_auroc,
            training_loss=train_loss
        )

# Replace the main training loop with this:
if __name__ == "__main__":
    # Configure the ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=NUM_EPOCHS,
        grace_period=5,
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        metric_columns=["loss", "auroc", "training_loss", "time_total_s"]
    )
    
    # Run the hyperparameter search
    analysis = tune.run(
        train_with_tune,
        resources_per_trial={"gpu": 1},
        config=config,
        num_samples=20,  # Number of parameter combinations to try
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./ray_tune_results",  # Directory to save results
        name="vuln_detection_tune"
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial validation AUROC: {best_trial.last_result['auroc']}")
    
    # Save best parameters
    best_params = best_trial.config
    torch.save(best_params, "best_hyperparams.pt")