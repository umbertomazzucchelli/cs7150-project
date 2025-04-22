# Human Activity Recognition (HAR) with LSTM and RNNs

A comprehensive comparison of machine learning and deep learning models for Human Activity Recognition using accelerometer data from wearable sensors.

## Project Overview

This project implements and compares multiple advanced machine learning approaches for Human Activity Recognition (HAR) using accelerometer data. The system processes time-series acceleration data collected from body-worn sensors to classify human activities such as walking, sitting, standing, and other physical movements.

### Datasets

The system works with two datasets:

1. **WISDM Dataset**:
   - 6 activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
   - Phone accelerometer data sampled at 20Hz
   - Single sensor placement (phone in pocket, treated as "thigh" location)

2. **PAAWS Dataset**:
   - 8 activities: Sitting, Standing, Walking, Walking Up Stairs, Walking Down Stairs, Biking, Gym Exercises, Lying Down
   - Wearable accelerometer data sampled at 80Hz
   - Two sensor placements: wrist and thigh

### Models Implemented

1. **LSTM Network** - Long Short-Term Memory neural network for sequence learning
2. **CNN-LSTM Hybrid** - Convolutional Neural Network combined with LSTM for feature extraction and sequential processing
3. **Transformer Network** - Self-attention based architecture for capturing long-range dependencies
4. **Random Forest** - Traditional machine learning approach with extensive feature engineering

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- tqdm

### Setup

1. Clone the repository
2. Install dependencies
3. Download the datasets (see Data Preparation section)

### Data Preparation

The raw data needs to be preprocessed before being used for training:

```bash
# Preprocess the WISDM dataset
python preprocess_wisdm.py
```

## Running the Models

### Running All Models (Comparison)

To run all models in sequence for comparison:

```bash
./run_har_comparison.sh
```

This script will:
1. Preprocess the WISDM dataset
2. Run all models (Random Forest, LSTM, CNN, Transformer) on both datasets

### Running Individual Models

You can also run individual models with specific parameters:

```bash
# Random Forest model
python HAR_Recognition_RandomForest.py --cv_type kfold --k_folds 5

# LSTM model
python HAR_Recognition_LSTM_CNN.py --model lstm --cv_type kfold --k_folds 5

# CNN-LSTM model
python HAR_Recognition_LSTM_CNN.py --model cnn --cv_type kfold --k_folds 5

# Transformer model
python HAR_Recognition_Transformer.py --cv_type kfold --k_folds 5
```

### Cross-Validation Options

All models support two types of cross-validation:
- **k-fold** (`--cv_type kfold`): Standard k-fold cross-validation (default k=5)
- **LOSO** (`--cv_type loso`): Leave-One-Subject-Out cross-validation, which tests generalization to new users

### Parallel Processing

For faster execution, specify the number of parallel workers:

```bash
python run_har_models.py --max_workers 12 --log_dir "logs_wisdm"
```

## Model Architecture Details

### LSTM Network
- Multi-layered LSTM with 2 layers and 64 hidden neurons
- Input: Time windows of 3-axis accelerometer data (800 samples for R1 and 200 for WISDM)
- Hyperparameters: 
  - Learning rate: 4e-4
  - L2 regularization: 1.5e-3 
  - Dropout: 0.1
  - Batch size: 32

### CNN-LSTM Hybrid
- Three convolutional layers followed by a bidirectional LSTM
- Conv layers: 32→64→128 filters with 5×1 kernels
- Final bidirectional LSTM with 64 hidden neurons

### Transformer Network
- Positional encoding + multi-head self-attention
- Embedding dimension: 64
- Number of heads: 4
- Number of encoder layers: 2
- Feedforward dimension: 128

### Random Forest
- Feature extraction: Time domain and frequency domain features
- 100 estimators
- Class-balanced weights
- Feature selection: sqrt(n_features)

## Results

The model logs and results are stored in:
- `logs/` - Default log directory
- `logs_wisdm/` - Logs for WISDM dataset runs
- `models/` - Saved model files
- `RESULTS/` - Detailed results and visualizations

## Project Structure

- `preprocess_wisdm.py` - Preprocesses the WISDM dataset
- `run_har_models.py` - Main script for running all models
- `run_har_comparison.sh` - Shell script for complete comparison workflow
- `HAR_Recognition_LSTM_CNN.py` - LSTM and CNN-LSTM implementations
- `HAR_Recognition_Transformer.py` - Transformer model implementation
- `HAR_Recognition_RandomForest.py` - Random Forest model with feature engineering
- `data_process.py` - Data processing utilities
- `params_optimization.py` - Hyperparameter optimization
