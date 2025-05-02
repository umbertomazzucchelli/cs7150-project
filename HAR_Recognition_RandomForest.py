import numpy as np
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
from tqdm import tqdm

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

# Model parameters (Shared for windowing)
SEGMENT_TIME_SIZE = 1024  # Default for R1 (80Hz * 10s = 800 samples)
# Adjust for WISDM's 20Hz sampling rate
if DATASET_NAME == "WISDM":
    SEGMENT_TIME_SIZE = 256  # 20Hz * 10s = 200 samples

OVERLAP = 0.5
TIME_STEP = int(SEGMENT_TIME_SIZE * (1 - OVERLAP / 100))


##################################################
### ARGUMENT PARSING (Unchanged)
##################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Human Activity Recognition using Random Forest")
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
### FEATURE EXTRACTION (Unchanged)
##################################################
def extract_features(segments):
    """
    Extract time and frequency domain features from each segment
    """
    n_segments = len(segments)
    feature_list = []

    for i in tqdm(range(n_segments), desc="Extracting Features", leave=False):
        segment = segments[i]
        features = []

        # For each axis (x, y, z)
        for axis in range(segment.shape[1]):
            axis_data = segment[:, axis]

            # Time domain features
            features.append(np.mean(axis_data))  # Mean
            features.append(np.std(axis_data))  # Standard deviation
            features.append(np.max(axis_data))  # Maximum
            features.append(np.min(axis_data))  # Minimum
            features.append(np.percentile(axis_data, 25))  # 25th percentile
            features.append(np.percentile(axis_data, 75))  # 75th percentile
            features.append(np.median(axis_data))  # Median
            features.append(np.sum(np.abs(axis_data)))  # Sum of absolute values

            # Calculate zero crossings
            zero_crossings = np.sum(np.diff(np.sign(axis_data)) != 0)
            features.append(zero_crossings)

            # Frequency domain features
            fft_values = np.abs(np.fft.fft(axis_data))
            fft_freq = np.fft.fftfreq(len(axis_data))

            # Only consider positive frequencies
            pos_mask = fft_freq > 0
            if np.any(pos_mask):
                fft_values = fft_values[pos_mask]
                fft_freq = fft_freq[pos_mask]
            else:
                # Handle cases with no positive frequencies (e.g., constant signal)
                fft_values = np.array([0.0])
                fft_freq = np.array([0.0])

            features.append(np.max(fft_values))  # Maximum frequency component
            features.append(np.mean(fft_values))  # Mean frequency
            features.append(np.std(fft_values))  # Std of frequency components

            # Energy of the signal (sum of squared FFT components)
            features.append(np.sum(fft_values**2))

            # Dominant frequency
            features.append(fft_freq[np.argmax(fft_values)])

        # Correlation between axes
        # Handle potential constant columns in segments causing NaN in corrcoef
        if np.std(segment[:, 0]) > 1e-6 and np.std(segment[:, 1]) > 1e-6:
            features.append(np.corrcoef(segment[:, 0], segment[:, 1])[0, 1])  # x-y correlation
        else:
            features.append(0.0)
        if np.std(segment[:, 0]) > 1e-6 and np.std(segment[:, 2]) > 1e-6:
            features.append(np.corrcoef(segment[:, 0], segment[:, 2])[0, 1])  # x-z correlation
        else:
            features.append(0.0)
        if np.std(segment[:, 1]) > 1e-6 and np.std(segment[:, 2]) > 1e-6:
            features.append(np.corrcoef(segment[:, 1], segment[:, 2])[0, 1])  # y-z correlation
        else:
            features.append(0.0)

        feature_list.append(features)

    return np.array(feature_list)


##################################################
### TRAIN AND EVALUATE FUNCTION (Minor change for N_CLASSES)
##################################################
def train_evaluate_fold(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model for a single fold
    """
    # Extract features
    print("Extracting training features...")
    X_train_features = extract_features(X_train)
    print("Extracting testing features...")
    X_test_features = extract_features(X_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"
    )

    print(f"Training Random Forest model with {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features...")
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    accuracy_val = accuracy_score(y_test, y_pred)
    f1_score_val = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    # Ensure confusion matrix has labels for all potential classes
    confusion_mat = confusion_matrix(y_test, y_pred, labels=range(N_CLASSES))

    return accuracy_val, f1_score_val, confusion_mat, model, scaler


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


def save_training_logs(results, confusion_matrix, timestamp, all_accuracies, all_f1_scores, n_folds, cv_type="LOSO"):
    log_dir = create_logs_directory()
    # Construct filename including dataset and location
    log_file_base = f"training_log_{timestamp}_rf_{DATASET_NAME}_{LOCATION}"
    log_file = os.path.join(log_dir, f"{log_file_base}.json")

    # Prepare the log data
    log_data = {
        "timestamp": timestamp,
        "dataset": DATASET_NAME,
        "location": LOCATION,
        "model_type": "random_forest",
        "hyperparameters": {
            "n_classes": N_CLASSES,
            "n_estimators": 100,
            "max_features": "sqrt",
            "random_state": RANDOM_SEED,
            "segment_time_size": SEGMENT_TIME_SIZE,
            "overlap": OVERLAP,
            "time_step": TIME_STEP,
            "cv_type": cv_type,
            "n_folds": n_folds,
        },
        "fold_results": {
            "accuracies": [float(acc) for acc in all_accuracies],
            "f1_scores": [float(f1) for f1 in all_f1_scores],
        },
        "final_results": {
            "average_accuracy": float(results["avg_accuracy"]),
            "std_accuracy": float(results["std_accuracy"]),
            "average_f1_score": float(results["avg_f1_score"]),
            "std_f1_score": float(results["std_f1_score"]),
            "confusion_matrix": confusion_matrix.tolist() if confusion_matrix is not None else None,
        },
    }

    # Save to JSON file
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)

    # Also save a human-readable log file
    txt_file = os.path.join(log_dir, f"{log_file_base}.txt")
    with open(txt_file, "w") as f:
        f.write(f"Random Forest Training Log - {timestamp}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Location: {LOCATION}\n")
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
    print(f"Using Random Forest model on {DATASET_NAME} dataset")
    print(f"Cross-validation type: {args.cv_type.upper()}")
    if args.cv_type == "kfold":
        print(f"Number of folds: {args.k_folds}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = create_logs_directory()  # Create directory structure early

    if args.train_full:
        model_dir = os.path.join(args.model_path, DATASET_NAME, LOCATION)
        os.makedirs(model_dir, exist_ok=True)

    np.random.seed(RANDOM_SEED)

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

        print("Extracting features from all data...")
        X_features = extract_features(X)

        try:
            X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
        except ValueError:
            print("Stratified split failed, using non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=RANDOM_SEED)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )

        print(f"Training Random Forest model with {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
        model.fit(X_train_scaled, y_train)

        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")

        model_type_str = "rf"
        model_save_path = os.path.join(model_dir, f"{model_type_str}_model_{timestamp}_{DATASET_NAME}_{LOCATION}.pkl")
        scaler_save_path = os.path.join(model_dir, f"{model_type_str}_scaler_{timestamp}_{DATASET_NAME}_{LOCATION}.pkl")

        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_save_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"\nModel saved to: {model_save_path}")
        print(f"Scaler saved to: {scaler_save_path}")

    else:
        # Cross-validation
        all_accuracies, all_f1_scores, all_confusion_matrices = [], [], []

        if args.cv_type == "loso":
            unique_subjects = np.unique(subject_ids_array)
            n_folds_cv = len(unique_subjects)
            print(f"Performing Leave-One-Subject-Out (LOSO) CV with {n_folds_cv} subjects...")

            for fold, subject_to_leave_out in tqdm(enumerate(unique_subjects, 1), total=n_folds_cv, desc="CV Progress"):
                print(f"\nFold {fold}/{n_folds_cv}: Leaving out Subject {subject_to_leave_out}")
                test_idx = np.where(subject_ids_array == subject_to_leave_out)[0]
                train_idx = np.where(subject_ids_array != subject_to_leave_out)[0]

                if len(test_idx) == 0 or len(train_idx) == 0:
                    print(f"Skipping fold {fold} due to insufficient data for subject {subject_to_leave_out}.")
                    continue

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"Skipping fold {fold} due to empty train/test set after split.")
                    continue

                fold_accuracy, fold_f1, fold_confusion_matrix, _, _ = train_evaluate_fold(X_train, y_train, X_test, y_test)

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

            for fold, (train_idx, test_idx) in tqdm(enumerate(skf.split(X, y), 1), total=n_folds_cv, desc="CV Progress"):
                print(f"\nFold {fold}/{n_folds_cv}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"Skipping fold {fold} due to empty train/test set after split.")
                    continue

                fold_accuracy, fold_f1, fold_confusion_matrix, _, _ = train_evaluate_fold(X_train, y_train, X_test, y_test)

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

            save_training_logs(
                results,
                avg_confusion_matrix,
                timestamp,
                all_accuracies,
                all_f1_scores,
                n_folds_cv,
                cv_type=cv_type_str,
            )

            # Plot average confusion matrix
            plt.figure(figsize=(max(8, N_CLASSES), max(6, N_CLASSES)))
            cm_sum = np.sum(avg_confusion_matrix, axis=1, keepdims=1)
            cm_norm = np.divide(avg_confusion_matrix, cm_sum, out=np.zeros_like(avg_confusion_matrix, dtype=float), where=cm_sum != 0)
            sns.heatmap(cm_norm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"RF Avg CM ({cv_type_str} - {DATASET_NAME} - {LOCATION})", fontsize=14)
            plt.ylabel("True label", fontsize=12)
            plt.xlabel("Predicted label", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f"confusion_matrix_{timestamp}_rf_{DATASET_NAME}_{LOCATION}.png"))
            plt.close()
