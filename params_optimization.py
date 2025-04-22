import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import copy
import itertools

##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = ["user", "activity", "timestamp", "x-axis", "y-axis", "z-axis"]

# LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
LABELS = ["Sitting", "Standing", "Walking", "Walking_Up_Stairs", "Walking_Down_Stairs", "Biking", "Gym_Exercises"]

DATA_PATH = "data/WISDM_ar_v1.1_raw.txt"
RANDOM_SEED = 13

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Model parameters
N_CLASSES = 7
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration
N_LSTM_LAYERS = 2
N_EPOCHS = 100  # Increased epochs since we have early stopping
L2_LOSS = 2e-4  # Adjusted L2 regularization
LEARNING_RATE = 1e-3  # Adjusted base learning rate
DROPOUT = 0.2  # Increased dropout for better regularization

# Determine if we're using WISDM or R1 data
DATASET_NAME = os.environ.get("DATASET_NAME", "R1").upper()
# Set window size based on sampling rate (20Hz for WISDM, 80Hz for R1)
if DATASET_NAME == "WISDM":
    SEGMENT_TIME_SIZE = 256  # 20Hz * 12.8s = 256 samples
else:
    SEGMENT_TIME_SIZE = 1024  # 80Hz * 12.8s = 1024 samples

N_HIDDEN_NEURONS = 64  # Increased hidden neurons
BATCH_SIZE = 64  # Increased batch size
OVERLAP = 0.5
TIME_STEP = int(SEGMENT_TIME_SIZE * (1 - OVERLAP / 100))

# Optimization parameters
EARLY_STOPPING_PATIENCE = 15  # Increased patience
MIN_DELTA = 1e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-5  # Added explicit weight decay

# Hyperparameter search space
HYPERPARAMETERS = {
    "n_lstm_layers": [2, 3],
    "n_hidden_neurons": [32, 64, 128],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [32, 64, 128],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "l2_loss": [1e-4, 2e-4, 5e-4],
}


##################################################
### DATASET CLASS
##################################################
class HARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


##################################################
### MODEL CLASS
##################################################
class LSTM_HAR(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTM_HAR, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


##################################################
### EARLY STOPPING CLASS
##################################################
class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_f1, model):
        if self.best_loss is None:
            self.best_loss = val_f1
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_f1 < self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_f1
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop


##################################################
### LEARNING RATE SCHEDULER
##################################################
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


##################################################
### EVALUATION FUNCTION
##################################################
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    f1 = metrics.f1_score(all_labels, all_predictions, average="weighted")

    return total_loss / len(data_loader), accuracy, f1


##################################################
### HYPERPARAMETER TUNING CLASS
##################################################
class HyperparameterTuning:
    def __init__(self, X, y, n_folds=10):
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.best_params = None
        self.best_score = 0
        self.results = []

    def generate_parameter_combinations(self):
        keys = HYPERPARAMETERS.keys()
        values = HYPERPARAMETERS.values()
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def train_and_evaluate(self, params):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(self.X), 1):
            print(f"\nFold {fold}/{self.n_folds}")

            # Split data
            X_train_val, X_test = self.X[train_val_idx], self.X[test_idx]
            y_train_val, y_test = self.y[train_val_idx], self.y[test_idx]

            train_size = int(0.9 * len(X_train_val))
            X_train, X_val = X_train_val[:train_size], X_train_val[train_size:]
            y_train, y_val = y_train_val[:train_size], y_train_val[train_size:]

            # Calculate class weights
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
            class_weights = torch.FloatTensor(class_weights).to(DEVICE)

            # Create data loaders
            train_dataset = HARDataset(X_train, y_train)
            val_dataset = HARDataset(X_val, y_val)
            test_dataset = HARDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

            # Initialize model
            model = LSTM_HAR(
                input_size=N_FEATURES, hidden_size=params["n_hidden_neurons"], num_layers=params["n_lstm_layers"], num_classes=N_CLASSES, dropout=params["dropout"]
            ).to(DEVICE)

            # Setup training
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"], betas=(0.9, 0.999), eps=1e-8)

            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=N_EPOCHS)

            # Train model
            _ = train_model(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, scheduler, params["l2_loss"])

            # Evaluate on test set
            _, _, test_f1 = evaluate_model(model, test_loader, criterion, DEVICE)
            fold_scores.append(test_f1)

            print(f"Fold {fold} Test F1-Score: {test_f1:.4f}")

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        return mean_score, std_score, fold_scores

    def run_tuning(self):
        param_combinations = self.generate_parameter_combinations()
        print(f"Total parameter combinations to try: {len(param_combinations)}")

        for params in tqdm(param_combinations, desc="Hyperparameter Tuning"):
            print("\nTrying parameters:", params)
            mean_score, std_score, fold_scores = self.train_and_evaluate(params)

            self.results.append({"params": params, "mean_f1": mean_score, "std_f1": std_score, "fold_scores": fold_scores})

            print(f"Mean F1-Score: {mean_score:.4f} ± {std_score:.4f}")

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
                print("New best parameters found!")

        return self.best_params, self.best_score, self.results


##################################################
### MODIFIED TRAINING FUNCTION
##################################################
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None, l2_loss=2e-4):
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": [], "learning_rates": []}

    early_stopping = EarlyStopping()
    best_val_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # L2 regularization
            l2_reg = torch.tensor(0.0).to(DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_loss * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = metrics.accuracy_score(train_labels, train_predictions)
        train_f1 = metrics.f1_score(train_labels, train_predictions, average="weighted")

        # Validation phase
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, DEVICE)

        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, WarmupCosineScheduler):
                scheduler.step(epoch)
            else:
                scheduler.step()

        # Early stopping
        if early_stopping(val_f1, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            model.load_state_dict(early_stopping.best_model)
            break

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


##################################################
### LOGGING FUNCTIONS
##################################################
def create_logs_directory():
    if not os.path.exists("logs"):
        os.makedirs("logs")


def save_training_logs(history, results, confusion_matrix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")

    # Prepare the log data
    log_data = {
        "timestamp": timestamp,
        "hyperparameters": {
            "n_classes": N_CLASSES,
            "n_features": N_FEATURES,
            "n_lstm_layers": N_LSTM_LAYERS,
            "n_epochs": N_EPOCHS,
            "l2_loss": L2_LOSS,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "segment_time_size": SEGMENT_TIME_SIZE,
            "n_hidden_neurons": N_HIDDEN_NEURONS,
            "batch_size": BATCH_SIZE,
            "device": str(DEVICE),
            "random_seed": RANDOM_SEED,
            "overlap": OVERLAP,
            "time_step": TIME_STEP,
        },
        "fold_results": {
            "accuracies": [float(acc) for acc in all_accuracies],
            "f1_scores": [float(f1) for f1 in all_f1_scores],
            "training_histories": [
                {"fold": i + 1, "train_loss": h["train_loss"], "train_acc": h["train_acc"], "val_loss": h["val_loss"], "val_acc": h["val_acc"]} for i, h in enumerate(all_histories)
            ],
        },
        "final_results": {
            "average_accuracy": float(results["avg_accuracy"]),
            "std_accuracy": float(results["std_accuracy"]),
            "average_f1_score": float(results["avg_f1_score"]),
            "std_f1_score": float(results["std_f1_score"]),
            "confusion_matrix": confusion_matrix.tolist(),
        },
    }

    # Save to JSON file
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)

    # Also save a human-readable log file
    txt_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    with open(txt_file, "w") as f:
        f.write(f"Training Log - {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Hyperparameters:\n")
        f.write("-" * 40 + "\n")
        for key, value in log_data["hyperparameters"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Fold Results:\n")
        f.write("-" * 40 + "\n")
        for i, (acc, f1) in enumerate(zip(log_data["fold_results"]["accuracies"], log_data["fold_results"]["f1_scores"]), 1):
            f.write(f"Fold {i}:\n")
            f.write(f"  Accuracy: {acc:.4f}\n")
            f.write(f"  F1-Score: {f1:.4f}\n")
            f.write("\n")

        f.write("Final Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Accuracy: {log_data['final_results']['average_accuracy']:.4f} ± {log_data['final_results']['std_accuracy']:.4f}\n")
        f.write(f"Average F1-Score: {log_data['final_results']['average_f1_score']:.4f} ± {log_data['final_results']['std_f1_score']:.4f}\n")

    print(f"\nTraining logs saved to: {log_file}")
    print(f"Human-readable log saved to: {txt_file}")


##################################################
### SAVE TUNING RESULTS
##################################################
def save_tuning_results(results, best_params, best_score):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "tuning_results"
    os.makedirs(results_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(results_dir, f"hyperparameter_tuning_{timestamp}.json")
    results_data = {
        "timestamp": timestamp,
        "best_params": best_params,
        "best_score": float(best_score),
        "all_results": [
            {"params": r["params"], "mean_f1": float(r["mean_f1"]), "std_f1": float(r["std_f1"]), "fold_scores": [float(s) for s in r["fold_scores"]]} for r in results
        ],
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=4)

    # Create visualization of results
    plt.figure(figsize=(15, 10))
    scores = [r["mean_f1"] for r in results]
    plt.plot(scores, "b-", label="Mean F1-Score")
    plt.fill_between(range(len(scores)), [r["mean_f1"] - r["std_f1"] for r in results], [r["mean_f1"] + r["std_f1"] for r in results], alpha=0.2)
    plt.xlabel("Parameter Combination")
    plt.ylabel("F1-Score")
    plt.title("Hyperparameter Tuning Results")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"tuning_plot_{timestamp}.png"))
    plt.close()

    print(f"\nTuning results saved to: {results_file}")


##################################################
### MAIN
##################################################
if __name__ == "__main__":
    # Create logs directory
    create_logs_directory()

    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    # Load and preprocess data
    data = np.load("data/R1_data.npy")
    data = data[~np.isnan(data).any(axis=1)]
    data = data[data[:, -1] != 7]

    # Process segments
    segments = []
    labels = []
    subjects = np.unique(data[:, 0])

    print("Processing data...")
    for user_id in tqdm(subjects, desc="Processing users"):
        user_data = data[data[:, 0] == user_id]

        for i in range(0, len(user_data) - SEGMENT_TIME_SIZE + 1, TIME_STEP):
            if i + SEGMENT_TIME_SIZE > len(user_data):
                continue

            segment = user_data[i : i + SEGMENT_TIME_SIZE, 1:4]

            if len(segment) == SEGMENT_TIME_SIZE:
                segments.append(segment)
                activity_window = user_data[i : i + SEGMENT_TIME_SIZE, 4]
                unique_activities, counts = np.unique(activity_window, return_counts=True)
                most_common_activity = unique_activities[np.argmax(counts)]
                labels.append(most_common_activity)

    X = np.array(segments, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print("Data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Perform hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    tuner = HyperparameterTuning(X, y, n_folds=10)
    best_params, best_score, results = tuner.run_tuning()

    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best F1-Score: {best_score:.4f}")

    # Save tuning results
    save_tuning_results(results, best_params, best_score)

    # Initialize lists to store results
    all_confusion_matrices = []
    all_accuracies = []
    all_f1_scores = []
    all_histories = []

    # Initialize 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    print("Performing 5-fold cross-validation...")

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/5")

        # Split data into train+val and test
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        train_size = int(0.9 * len(X_train_val))
        X_train, X_val = X_train_val[:train_size], X_train_val[train_size:]
        y_train, y_val = y_train_val[:train_size], y_train_val[train_size:]

        # Calculate class weights for the training set
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)

        print("Class weights:", {LABELS[i]: f"{w:.3f}" for i, w in enumerate(class_weights.cpu().numpy())})
        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")

        # Create data loaders
        train_dataset = HARDataset(X_train, y_train)
        val_dataset = HARDataset(X_val, y_val)
        test_dataset = HARDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize the model
        model = LSTM_HAR(
            input_size=N_FEATURES,
            hidden_size=N_HIDDEN_NEURONS,
            num_layers=N_LSTM_LAYERS,
            num_classes=N_CLASSES,
            dropout=DROPOUT,
        ).to(DEVICE)

        # Loss with class weights and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)

        # Custom learning rate scheduler with warmup
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=N_EPOCHS)

        # Train the model using validation data for monitoring
        history = train_model(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, scheduler)
        all_histories.append(history)

        # Final evaluation on test set
        model.eval()
        test_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Calculate metrics for this fold using test set
        confusion_matrix = metrics.confusion_matrix(all_labels, all_predictions)
        accuracy = metrics.accuracy_score(all_labels, all_predictions)
        f1 = metrics.f1_score(all_labels, all_predictions, average="weighted")
        test_loss = test_loss / len(test_loader)

        all_confusion_matrices.append(confusion_matrix)
        all_accuracies.append(accuracy)
        all_f1_scores.append(f1)

        print(f"Fold {fold} Final Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1-Score: {f1:.4f}")

    # Calculate and display average results
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    avg_accuracy = np.mean(all_accuracies)
    avg_f1_score = np.mean(all_f1_scores)
    std_accuracy = np.std(all_accuracies)
    std_f1_score = np.std(all_f1_scores)

    results = {"avg_accuracy": avg_accuracy, "std_accuracy": std_accuracy, "avg_f1_score": avg_f1_score, "std_f1_score": std_f1_score}

    print("\n5-Fold Cross-Validation Results:")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average F1-Score: {avg_f1_score:.4f} ± {std_f1_score:.4f}")

    # Save training logs
    save_training_logs(history, results, avg_confusion_matrix)

    # Plot average confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        avg_confusion_matrix / np.sum(avg_confusion_matrix, axis=1, keepdims=1),
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot=True,
    )
    plt.title("Average Confusion Matrix (5-Fold CV)")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("avg_confusion_matrix_pytorch.png")
    plt.close()

    # Plot final training history
    plt.figure(figsize=(12, 8))
    plt.plot(history["train_loss"], "r--", label="Train loss")
    plt.plot(history["train_acc"], "b--", label="Train accuracy")
    plt.plot(history["val_loss"], "r-", label="Val loss")
    plt.plot(history["val_acc"], "b-", label="Val accuracy")
    plt.title("Final Training Session Progress")
    plt.legend(loc="upper right", shadow=True)
    plt.ylabel("Training Progress (Loss or Accuracy values)")
    plt.xlabel("Training Epoch")
    plt.ylim(0)
    plt.savefig("final_error_plot_pytorch.png")
    plt.close()

    # # Plot learning rate
    # plt.figure(figsize=(10, 6))
    # plt.plot(history["learning_rates"], "g-", linewidth=2)
    # plt.title("Learning Rate Schedule")
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Training Epoch")
    # plt.grid(True)
    # plt.savefig("learning_rate_schedule.png")
    # plt.close()
