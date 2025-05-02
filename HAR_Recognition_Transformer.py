import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse
from sklearn.preprocessing import StandardScaler
import pickle
import math

##################################################
### DATASET SPECIFIC PARAMETERS
##################################################
DATASET_PARAMS = {
    "R1": {
        "LABELS": ["Sitting", "Standing", "Walking", "Walking_Up_Stairs", "Walking_Down_Stairs", "Biking", "Gym_Exercises", "Lying_Down"],
        "N_CLASSES": 8,
        "data_file_pattern": "data/combined/R1_{location}_data.npy",
    },
    "WISDM": {
        "LABELS": ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"],
        "N_CLASSES": 6,
        "data_file_pattern": "data/combined/WISDM_data.npy",  # Location not used
    },
}

# Determine dataset from environment variable, default to R1
DATASET_NAME = os.environ.get("DATASET_NAME", "R1").upper()
if DATASET_NAME not in DATASET_PARAMS:
    raise ValueError(f"Invalid DATASET_NAME: {DATASET_NAME}. Choose from {list(DATASET_PARAMS.keys())}")

# Get location from environment variable
LOCATION = os.environ.get("LOCATION", "thigh").lower()

# Get custom log directory from environment if available
CUSTOM_LOG_DIR = os.environ.get("LOG_DIR", "logs")

# Set dataset-specific globals
CURRENT_DATASET_PARAMS = DATASET_PARAMS[DATASET_NAME]
LABELS = CURRENT_DATASET_PARAMS["LABELS"]
N_CLASSES = CURRENT_DATASET_PARAMS["N_CLASSES"]
DATA_FILE_PATTERN = CURRENT_DATASET_PARAMS["data_file_pattern"]
print(f"Using Dataset: {DATASET_NAME} with {N_CLASSES} classes: {LABELS}")
print(f"Location: {LOCATION}")
print(f"Log directory: {CUSTOM_LOG_DIR}")


##################################################
### GLOBAL VARIABLES (Remaining)
##################################################
RANDOM_SEED = 13

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Data parameters (Shared)
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration
SEGMENT_TIME_SIZE = 1024  # Default for R1 (80Hz * 10s = 800 samples)
# Adjust for WISDM's 20Hz sampling rate
if DATASET_NAME == "WISDM":
    SEGMENT_TIME_SIZE = 256  # 20Hz * 10s = 200 samples
OVERLAP = 0.5
TIME_STEP = int(SEGMENT_TIME_SIZE * (1 - OVERLAP / 100))

# Default location - will be overridden by env var if provided
LOCATION = os.environ.get("LOCATION", "wrist" if DATASET_NAME == "R1" else "phone")
print(f"Using Location: {LOCATION}")

# Transformer Model parameters (Shared)
N_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
L2_LOSS = 1e-4
DROPOUT = 0.1
EMBED_DIM = 64  # Embedding dimension (must be divisible by N_HEADS)
N_HEADS = 4  # Number of attention heads
N_ENCODER_LAYERS = 2  # Number of Transformer encoder layers
FF_DIM = 128  # Dimension of the feedforward network model in nn.TransformerEncoderLayer

# CNN hyperparameters (for CNN+Transformer)
CNN_FILTERS = 32
CNN_KERNEL_SIZE = 5
POOL_SIZE = 4


##################################################
### DATASET CLASS (Unchanged)
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
### ARGUMENT PARSING (Unchanged)
##################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Human Activity Recognition using Transformer")
    parser.add_argument(
        "--train_full",
        action="store_true",
        help="Train on full dataset and export model for inference",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--cv_type",
        type=str,
        choices=["loso", "kfold"],
        default="loso",
        help="Cross-validation type: loso (Leave-One-Subject-Out) or kfold (K-Fold with k=5)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross-validation (default: 5)",
    )
    return parser.parse_args()


##################################################
### POSITIONAL ENCODING (Unchanged)
##################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


##################################################
### TRANSFORMER MODEL CLASS (Unchanged, uses global N_CLASSES)
##################################################
class Transformer_HAR(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_encoder_layers, ff_dim, num_classes, seq_len, dropout=0.1, 
                 cnn_filters=32, cnn_kernel_size=5, pool_size=4):
        super(Transformer_HAR, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.pool_size = pool_size

        # CNN Block
        # Input: (Batch, Features, SeqLen)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=cnn_kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        
        # Calculate sequence length after pooling
        # Input to pool: (Batch, cnn_filters, SeqLen)
        # Output of pool: (Batch, cnn_filters, floor(SeqLen / pool_size))
        self.new_seq_len = seq_len // pool_size # Use integer division

        # Embedding layer now takes CNN output channels as input dim
        self.input_embedding = nn.Linear(cnn_filters, embed_dim)
        
        # Positional Encoding needs the new sequence length
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=self.new_seq_len + 1) 
        
        # Transformer Encoder Layer (batch_first=True means input/output: Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embed_dim, num_classes)  # Uses num_classes passed during init

    def forward(self, x):
        # Input x shape: (Batch, SeqLen, Features)
        
        # CNN Block
        x = x.permute(0, 2, 1)  # -> (Batch, Features, SeqLen)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)       # -> (Batch, cnn_filters, new_seq_len)
        x = x.permute(0, 2, 1)  # -> (Batch, new_seq_len, cnn_filters)

        # Transformer Block
        x = self.input_embedding(x) * math.sqrt(self.embed_dim) # Input embedding now expects cnn_filters
        # Input to pos_encoder (and TransformerEncoderLayer) needs to be permuted if batch_first=False, but we use batch_first=True
        # So, keep as (Batch, new_seq_len, embed_dim)
        # x = x.permute(1, 0, 2) # No longer needed for batch_first=True
        x = self.pos_encoder(x) # Apply positional encoding directly
        # x = x.permute(1, 0, 2) # No longer needed for batch_first=True
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Global Average Pooling over the sequence dimension
        out = self.fc(x)
        return out


##################################################
### TRAINING FUNCTION (Unchanged)
##################################################
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None):
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": [], "learning_rates": []}
    patience = 20
    best_val_f1 = 0
    counter = 0
    early_stop = False
    best_model_state = None

    for epoch in tqdm(range(num_epochs), desc="Training Progress", total=num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_preds, train_true = [], []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            l2_reg = torch.tensor(0.0).to(DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += L2_LOSS * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        train_f1 = metrics.f1_score(train_true, train_preds, average="weighted", zero_division=0)
        val_f1 = metrics.f1_score(val_true, val_preds, average="weighted", zero_division=0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs - no improvement in validation F1 score")
                early_stop = True
                break
        if early_stop:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation F1 score: {best_val_f1:.4f}")

    return history


##################################################
### FOLD TRAINING AND EVALUATION (Minor changes for N_CLASSES)
##################################################
def train_evaluate_fold(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate the Transformer model for a single fold
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    unique_classes = np.unique(y_train)
    class_weights = np.zeros(N_CLASSES)
    present_classes_mask = np.isin(np.arange(N_CLASSES), unique_classes)
    if np.any(present_classes_mask):
        weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
        for idx, class_idx in enumerate(unique_classes):
            if class_idx < N_CLASSES:
                class_weights[class_idx] = weights[idx]
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print("Class weights (Fold):", {LABELS[i]: f"{w:.3f}" for i, w in enumerate(class_weights.cpu().numpy()) if w > 0})
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    train_dataset = HARDataset(X_train, y_train)
    val_dataset = HARDataset(X_val, y_val)
    test_dataset = HARDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the Transformer model (using global N_CLASSES)
    model = Transformer_HAR(
        input_dim=N_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=N_HEADS,
        num_encoder_layers=N_ENCODER_LAYERS,
        ff_dim=FF_DIM,
        num_classes=N_CLASSES,  # Pass the correct N_CLASSES
        seq_len=SEGMENT_TIME_SIZE,
        dropout=DROPOUT,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_size=CNN_KERNEL_SIZE,
        pool_size=POOL_SIZE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, scheduler)

    model.eval()
    test_loss = 0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Ensure confusion matrix covers all N_CLASSES
    confusion_mat = metrics.confusion_matrix(all_labels, all_predictions, labels=range(N_CLASSES))
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    f1 = metrics.f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    return history, accuracy, f1, confusion_mat, model, scaler


##################################################
### LOGGING FUNCTIONS (Add DATASET_NAME and location handling)
##################################################
def create_logs_directory():
    # Creates logs/DATASET_NAME/LOCATION structure
    log_base_dir = os.path.join(CUSTOM_LOG_DIR, DATASET_NAME.lower())
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)
    log_dir = os.path.join(log_base_dir, LOCATION)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_training_logs(results, confusion_matrix, timestamp, all_accuracies, all_f1_scores, all_histories, n_folds, cv_type="LOSO"):
    log_dir = create_logs_directory()
    # Construct filename including dataset and location
    log_file_base = f"training_log_{timestamp}_transformer_{DATASET_NAME}_{LOCATION}"
    log_file = os.path.join(log_dir, f"{log_file_base}.json")

    log_data = {
        "timestamp": timestamp,
        "dataset": DATASET_NAME,
        "location": LOCATION,
        "model_type": "transformer",
        "hyperparameters": {
            "n_classes": N_CLASSES,
            "n_features": N_FEATURES,
            "n_encoder_layers": N_ENCODER_LAYERS,
            "n_heads": N_HEADS,
            "embed_dim": EMBED_DIM,
            "ff_dim": FF_DIM,
            "n_epochs": N_EPOCHS,
            "l2_loss": L2_LOSS,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "segment_time_size": SEGMENT_TIME_SIZE,
            "batch_size": BATCH_SIZE,
            "device": str(DEVICE),
            "random_seed": RANDOM_SEED,
            "overlap": OVERLAP,
            "time_step": TIME_STEP,
            "cv_type": cv_type,
            "n_folds": n_folds,
            "early_stopping_patience": 20,
            "early_stopping_metric": "validation_f1_score",
        },
        "fold_results": {
            "accuracies": [float(acc) for acc in all_accuracies],
            "f1_scores": [float(f1) for f1 in all_f1_scores],
            "training_histories": [
                {
                    "fold": i + 1,
                    "train_loss": h["train_loss"],
                    "train_acc": h["train_acc"],
                    "train_f1": h["train_f1"],
                    "val_loss": h["val_loss"],
                    "val_acc": h["val_acc"],
                    "val_f1": h["val_f1"],
                }
                for i, h in enumerate(all_histories)
            ],
        },
        "final_results": {
            "average_accuracy": float(results["avg_accuracy"]),
            "std_accuracy": float(results["std_accuracy"]),
            "average_f1_score": float(results["avg_f1_score"]),
            "std_f1_score": float(results["std_f1_score"]),
            "confusion_matrix": confusion_matrix.tolist() if confusion_matrix is not None else None,
        },
    }

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)

    txt_file = os.path.join(log_dir, f"{log_file_base}.txt")
    with open(txt_file, "w") as f:
        f.write(f"Training Log - {timestamp}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Location: {LOCATION}\n")
        f.write("Model Type: Transformer\n")
        f.write(f"CV Type: {log_data['hyperparameters']['cv_type']} ({log_data['hyperparameters']['n_folds']} folds)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Hyperparameters:\n")
        f.write("-" * 40 + "\n")
        for key, value in log_data["hyperparameters"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Fold Results:\n")
        f.write("-" * 40 + "\n")
        for i, (acc, f1) in enumerate(zip(log_data["fold_results"]["accuracies"], log_data["fold_results"]["f1_scores"]), 1):
            f.write(f"Fold {i}: Accuracy: {acc:.4f}, F1-Score: {f1:.4f}\n")
        f.write("\n")
        f.write("Final Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Accuracy: {log_data['final_results']['average_accuracy']:.4f} ± {log_data['final_results']['std_accuracy']:.4f}\n")
        f.write(f"Average F1-Score: {log_data['final_results']['average_f1_score']:.4f} ± {log_data['final_results']['std_f1_score']:.4f}\n")

    print(f"\nTraining logs saved to: {log_file}")
    print(f"Human-readable log saved to: {txt_file}")


##################################################
### MAIN (Adapted for Dataset Handling)
##################################################
if __name__ == "__main__":
    args = parse_arguments()
    print(f"Using Transformer model architecture on {DATASET_NAME} dataset")
    print(f"Cross-validation type: {args.cv_type.upper()}")
    if args.cv_type == "kfold":
        print(f"Number of folds: {args.k_folds}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = create_logs_directory()  # Create directory structure early

    if args.train_full:
        model_dir = os.path.join(args.model_path, DATASET_NAME, LOCATION)
        os.makedirs(model_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    # Construct data path based on dataset and location
    data_path = DATA_FILE_PATTERN.format(location=LOCATION)
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Ensure the corresponding preprocessing script (e.g., preprocess_wisdm.py) has been run.")
    # Data format: [user, x, y, z, activity_code]
    data = np.load(data_path)
    data = data[~np.isnan(data).any(axis=1)]
    max_label = N_CLASSES - 1
    data = data[data[:, -1] <= max_label]

    segments, labels_list, segment_subjects = [], [], []
    subjects = np.unique(data[:, 0])
    n_subjects = len(subjects)
    if n_subjects == 0:
        raise ValueError("No subjects found")

    print("Starting data windowing...")
    for user_id in subjects:
        # if DATASET_NAME == "R1" and user_id == 10: continue
        user_data = data[data[:, 0] == user_id]
        for i in range(0, len(user_data) - SEGMENT_TIME_SIZE + 1, TIME_STEP):
            if i + SEGMENT_TIME_SIZE <= len(user_data):
                segment = user_data[i : i + SEGMENT_TIME_SIZE, 1:4]
                if len(segment) == SEGMENT_TIME_SIZE:
                    segments.append(segment)
                    segment_subjects.append(user_id)
                    activity_window = user_data[i : i + SEGMENT_TIME_SIZE, -1]
                    unique_acts, counts = np.unique(activity_window, return_counts=True)
                    labels_list.append(unique_acts[np.argmax(counts)])

    X = np.array(segments, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    subject_ids_array = np.array(segment_subjects)

    print(f"Windowing complete. X shape: {X.shape}, y shape: {y.shape}, Subjects shape: {subject_ids_array.shape}")
    print(f"Unique labels in data: {np.unique(y)}")
    if len(X) == 0:
        raise ValueError("No segments created after windowing. Check data and parameters.")

    if args.train_full:
        print("\nTraining on full dataset...")
        # Use stratified split if possible
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y)
        except ValueError:
            print("Stratified split failed, using non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RANDOM_SEED)

        print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        unique_classes_train = np.unique(y_train)
        class_weights = np.zeros(N_CLASSES)
        present_classes_mask_train = np.isin(np.arange(N_CLASSES), unique_classes_train)
        if np.any(present_classes_mask_train):
            weights_train = compute_class_weight(class_weight="balanced", classes=unique_classes_train, y=y_train)
            for idx, class_idx in enumerate(unique_classes_train):
                if class_idx < N_CLASSES:
                    class_weights[class_idx] = weights_train[idx]
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)

        train_dataset = HARDataset(X_train, y_train)
        val_dataset = HARDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = Transformer_HAR(
            input_dim=N_FEATURES,
            embed_dim=EMBED_DIM,
            num_heads=N_HEADS,
            num_encoder_layers=N_ENCODER_LAYERS,
            ff_dim=FF_DIM,
            num_classes=N_CLASSES,
            seq_len=SEGMENT_TIME_SIZE,
            dropout=DROPOUT,
            cnn_filters=CNN_FILTERS,
            cnn_kernel_size=CNN_KERNEL_SIZE,
            pool_size=POOL_SIZE,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

        print("Training final model...")
        history = train_model(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, scheduler)

        model_type_str = "transformer"
        model_save_path = os.path.join(model_dir, f"{model_type_str}_model_{timestamp}_{DATASET_NAME}_{LOCATION}.pt")
        scaler_save_path = os.path.join(model_dir, f"{model_type_str}_scaler_{timestamp}_{DATASET_NAME}_{LOCATION}.pkl")

        model_info = {
            "state_dict": model.state_dict(),
            "architecture": "transformer",
            "hyperparameters": {
                "n_features": N_FEATURES,
                "embed_dim": EMBED_DIM,
                "n_heads": N_HEADS,
                "n_encoder_layers": N_ENCODER_LAYERS,
                "ff_dim": FF_DIM,
                "n_classes": N_CLASSES,
                "dropout": DROPOUT,
                "segment_time_size": SEGMENT_TIME_SIZE,
            },
            "dataset_name": DATASET_NAME,
            "location": LOCATION,
        }
        torch.save(model_info, model_save_path)
        with open(scaler_save_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"\nModel saved to: {model_save_path}")
        print(f"Scaler saved to: {scaler_save_path}")

        plt.figure(figsize=(12, 8))
        plt.plot(history["train_loss"], "r--", label="Train loss")
        plt.plot(history["train_acc"], "b--", label="Train accuracy")
        plt.plot(history["val_loss"], "r-", label="Val loss")
        plt.plot(history["val_acc"], "b-", label="Val accuracy")
        plt.title(f"Training Progress (Full Dataset: {DATASET_NAME} - {LOCATION})")
        plt.legend(loc="upper right", shadow=True)
        plt.ylabel("Loss/Accuracy")
        plt.xlabel("Epoch")
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, f"training_full_{model_type_str}_{timestamp}_{DATASET_NAME}_{LOCATION}.png"))
        plt.close()

    else:
        # Cross-validation
        all_accuracies, all_f1_scores, all_histories, all_confusion_matrices = [], [], [], []

        if args.cv_type == "loso":
            unique_subjects = np.unique(subject_ids_array)
            n_folds_cv = len(unique_subjects)
            print(f"Performing Leave-One-Subject-Out (LOSO) CV with {n_folds_cv} subjects...")

            for fold, subject_to_leave_out in tqdm(enumerate(unique_subjects, 1), total=n_folds_cv, desc="CV Progress"):
                print(f"\nFold {fold}/{n_folds_cv}: Leaving out Subject {subject_to_leave_out}")
                test_idx = np.where(subject_ids_array == subject_to_leave_out)[0]
                train_val_idx = np.where(subject_ids_array != subject_to_leave_out)[0]

                if len(test_idx) == 0 or len(train_val_idx) == 0:
                    print(f"Skipping fold {fold} due to insufficient data for subject {subject_to_leave_out}.")
                    continue

                X_train_val, X_test = X[train_val_idx], X[test_idx]
                y_train_val, y_test = y[train_val_idx], y[test_idx]

                val_size = max(1, int(0.1 * len(X_train_val)))
                if len(X_train_val) > 1:
                    try:
                        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=RANDOM_SEED, stratify=y_train_val)
                    except ValueError:
                        print(f"Fold {fold}: Stratified validation split failed, using non-stratified.")
                        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=RANDOM_SEED)
                else:
                    X_train, y_train = X_train_val, y_train_val
                    X_val, y_val = X_train_val, y_train_val

                if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                    print(f"Skipping fold {fold} due to empty train/val/test set after split.")
                    continue

                fold_history, fold_accuracy, fold_f1, fold_confusion_matrix, _, _ = train_evaluate_fold(X_train, y_train, X_val, y_val, X_test, y_test)

                all_histories.append(fold_history)
                all_accuracies.append(fold_accuracy)
                all_f1_scores.append(fold_f1)
                all_confusion_matrices.append(fold_confusion_matrix)

                print(f"Fold {fold} Test Results: Accuracy: {fold_accuracy:.4f}, F1-Score: {fold_f1:.4f}")

            cv_type_str = "LOSO"

        else:  # kfold
            from sklearn.model_selection import StratifiedKFold

            n_folds_cv = args.k_folds
            print(f"Performing {n_folds_cv}-Fold Stratified CV...")
            skf = StratifiedKFold(n_splits=n_folds_cv, shuffle=True, random_state=RANDOM_SEED)

            for fold, (train_val_idx, test_idx) in tqdm(enumerate(skf.split(X, y), 1), total=n_folds_cv, desc="CV Progress"):
                print(f"\nFold {fold}/{n_folds_cv}")
                X_train_val, X_test = X[train_val_idx], X[test_idx]
                y_train_val, y_test = y[train_val_idx], y[test_idx]

                val_size = max(1, int(0.1 * len(X_train_val)))
                if len(X_train_val) > 1:
                    try:
                        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=RANDOM_SEED, stratify=y_train_val)
                    except ValueError:
                        print(f"Fold {fold}: Stratified validation split failed, using non-stratified.")
                        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=RANDOM_SEED)
                else:
                    X_train, y_train = X_train_val, y_train_val
                    X_val, y_val = X_train_val, y_train_val

                if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                    print(f"Skipping fold {fold} due to empty train/val/test set after split.")
                    continue

                fold_history, fold_accuracy, fold_f1, fold_confusion_matrix, _, _ = train_evaluate_fold(X_train, y_train, X_val, y_val, X_test, y_test)

                all_histories.append(fold_history)
                all_accuracies.append(fold_accuracy)
                all_f1_scores.append(fold_f1)
                all_confusion_matrices.append(fold_confusion_matrix)

                print(f"Fold {fold} Test Results: Accuracy: {fold_accuracy:.4f}, F1-Score: {fold_f1:.4f}")

            cv_type_str = f"{n_folds_cv}-Fold"

        # Calculate and display average results
        if not all_accuracies:
            print("\nNo folds were successfully completed. Cannot calculate average results.")
        else:
            avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
            avg_accuracy = np.mean(all_accuracies)
            avg_f1_score = np.mean(all_f1_scores)
            std_accuracy = np.std(all_accuracies)
            std_f1_score = np.std(all_f1_scores)

            results = {
                "avg_accuracy": avg_accuracy,
                "std_accuracy": std_accuracy,
                "avg_f1_score": avg_f1_score,
                "std_f1_score": std_f1_score,
            }

            print(f"\n{cv_type_str} CV Results ({n_folds_cv} folds performed on {DATASET_NAME} - {LOCATION}):")
            print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"Average F1-Score: {avg_f1_score:.4f} ± {std_f1_score:.4f}")

            save_training_logs(results, avg_confusion_matrix, timestamp, all_accuracies, all_f1_scores, all_histories, n_folds_cv, cv_type=cv_type_str)

            # Plot average confusion matrix
            if avg_confusion_matrix is not None:
                plt.figure(figsize=(max(8, N_CLASSES), max(6, N_CLASSES)))
                cm_sum = np.sum(avg_confusion_matrix, axis=1, keepdims=1)
                cm_norm = np.divide(avg_confusion_matrix, cm_sum, out=np.zeros_like(avg_confusion_matrix, dtype=float), where=cm_sum != 0)
                sns.heatmap(cm_norm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=".2f", cmap="viridis")
                plt.title(f"Transformer Avg CM ({cv_type_str} - {DATASET_NAME} - {LOCATION})", fontsize=14)
                plt.ylabel("True label", fontsize=12)
                plt.xlabel("Predicted label", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, f"confusion_matrix_{timestamp}_transformer_{DATASET_NAME}_{LOCATION}.png"))
                plt.close()
            else:
                print("Could not plot confusion matrix (no results).")
