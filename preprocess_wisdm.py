import numpy as np
import pandas as pd
import os

# Define standard column names and target activities for WISDM v1.1
COLUMN_NAMES = ["user", "activity", "timestamp", "x-axis", "y-axis", "z-axis"]
# Activities from the WISDM paper (Kwapisz et al., 2010)
WISDM_ACTIVITIES = [
    "Walking",
    "Jogging",
    "Upstairs",  # Corresponds to 'Walking_Up_Stairs' in R1
    "Downstairs",  # Corresponds to 'Walking_Down_Stairs' in R1
    "Sitting",
    "Standing",
]

# Mapping from activity name to integer code
ACTIVITY_MAPPING = {activity: i for i, activity in enumerate(WISDM_ACTIVITIES)}

# Input raw data file path
RAW_DATA_FILE = "wisdm/WISDM_ar_v1.1_raw.txt"
# Output directory and file path
OUTPUT_DIR = "data/combined"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "WISDM_data.npy")


def preprocess_wisdm_data(raw_file_path, output_file_path):
    """
    Reads the raw WISDM text file, preprocesses it, and saves relevant columns to a .npy file.
    Handles potential trailing semicolons and filters for target activities.
    """
    print(f"Starting preprocessing of WISDM raw data: {raw_file_path}")

    # Check if raw file exists
    if not os.path.exists(raw_file_path):
        print(f"Error: Raw data file not found at {raw_file_path}")
        print("Please ensure the WISDM dataset (v1.1) is downloaded and placed in the 'wisdm' directory.")
        return False

    # Read the data, skipping bad lines and handling potential errors
    try:
        df = pd.read_csv(raw_file_path, header=None, names=COLUMN_NAMES, on_bad_lines="skip")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    print(f"Read {len(df)} lines initially.")

    # Data Cleaning:
    # 1. Remove potential trailing semicolons from the 'z-axis' column if they exist
    #    Check if the column is object type first
    if df["z-axis"].dtype == "object":
        df["z-axis"] = df["z-axis"].astype(str).str.rstrip(";")
        # Attempt to convert columns to numeric, coercing errors
        for col in ["x-axis", "y-axis", "z-axis"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Drop rows with NaN values that might have resulted from coercion or were present initially
    initial_rows = len(df)
    df.dropna(subset=["user", "activity", "timestamp", "x-axis", "y-axis", "z-axis"], inplace=True)
    rows_after_na = len(df)
    if initial_rows > rows_after_na:
        print(f"Dropped {initial_rows - rows_after_na} rows due to missing/invalid numeric values.")

    # 3. Convert relevant columns to appropriate types
    df["user"] = df["user"].astype(int)
    df["timestamp"] = df["timestamp"].astype(np.int64)  # Ensure timestamp is integer
    df["x-axis"] = df["x-axis"].astype(float)
    df["y-axis"] = df["y-axis"].astype(float)
    df["z-axis"] = df["z-axis"].astype(float)

    # Filter for the target activities
    initial_rows = len(df)
    df = df[df["activity"].isin(WISDM_ACTIVITIES)]
    rows_after_filter = len(df)
    if initial_rows > rows_after_filter:
        print(f"Filtered {initial_rows - rows_after_filter} rows for activities not in {WISDM_ACTIVITIES}.")

    if rows_after_filter == 0:
        print("Error: No data remaining after filtering for target activities. Check activity names.")
        return False

    # Map activity names to integer codes
    df["activity_code"] = df["activity"].map(ACTIVITY_MAPPING)

    # Select and order the columns for the output .npy file
    # Format: [user, x, y, z, activity_code]
    output_df = df[["user", "x-axis", "y-axis", "z-axis", "activity_code"]].copy()

    # Convert DataFrame to NumPy array
    data_array = output_df.to_numpy(dtype=np.float32)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save the NumPy array
    np.save(output_file_path, data_array)
    print(f"Preprocessing complete. Saved {len(data_array)} samples to {output_file_path}")
    print(f"Data shape: {data_array.shape}")
    print(f"Unique users: {len(np.unique(data_array[:, 0]))}")
    print(f"Unique activity codes: {np.unique(data_array[:, -1])}")
    return True


if __name__ == "__main__":
    if not preprocess_wisdm_data(RAW_DATA_FILE, OUTPUT_FILE):
        print("WISDM data preprocessing failed.")
        exit(1)
    else:
        print("WISDM data preprocessing successful.")
