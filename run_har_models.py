#!/usr/bin/env python3

import subprocess
import time
import os
import argparse
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def run_model(model_type, dataset_name, location, log_dir_base):
    """
    Run the HAR recognition model with the specified architecture, dataset, and body location.

    Args:
        model_type (str): The model architecture to use ('lstm', 'cnn', 'rf', or 'transformer').
        dataset_name (str): The dataset to use ('WISDM' or 'R1').
        location (str): The body location of the sensor ('wrist', 'thigh' for R1, or 'thigh' for WISDM).
        log_dir_base (str): Base directory for logs.

    Returns:
        tuple: (config_name, success, time_taken)
    """
    config_name = f"{model_type}_{dataset_name}_{location}"

    # Create a log file for this specific run
    log_dir = os.path.join(log_dir_base, dataset_name.lower(), location.lower())
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_type.lower()}_run.log")

    with open(log_file, "w") as f:
        f.write(f"Running {model_type.upper()} model on {dataset_name.upper()} dataset (Location: {location.upper()})\n")
        f.write(f"{'=' * 80}\n")

        # Determine the script and command based on model_type
        script_parts = ""
        command = []
        if model_type == "rf":
            script_parts = "HAR_Recognition_RandomForest.py --cv_type kfold --k_folds 5".split()
            command = ["python"] + script_parts
        elif model_type == "transformer":
            script_parts = "HAR_Recognition_Transformer.py --cv_type kfold --k_folds 5".split()
            command = ["python"] + script_parts
        elif model_type in ["lstm", "cnn"]:
            script_parts = "HAR_Recognition_LSTM_CNN.py --cv_type kfold --k_folds 5".split()
            command = ["python"] + script_parts
        else:
            error_msg = f"Error: Unknown model type '{model_type}'"
            f.write(f"{error_msg}\n")
            return config_name, False, 0

        # Set environment variables for the recognition script
        env = os.environ.copy()
        env["DATASET_NAME"] = dataset_name
        env["LOCATION"] = location
        env["LOG_DIR"] = log_dir_base  # Pass the log directory to the HAR scripts

        # Log the command correctly
        f.write(f"Executing command: {' '.join(command)} with DATASET_NAME={dataset_name} LOCATION={location} LOG_DIR={log_dir_base}\n")
        f.flush()

    # Execute the command
    try:
        start_time = time.time()

        # Redirect stdout and stderr to log file
        with open(log_file, "a") as f:
            subprocess.run(command, check=True, env=env, stdout=f, stderr=f)

        end_time = time.time()
        time_taken = (end_time - start_time) / 60

        with open(log_file, "a") as f:
            f.write(f"\nCompleted {model_type.upper()} model on {dataset_name.upper()} dataset (Location: {location.upper()})\n")
            f.write(f"Time taken: {time_taken:.2f} minutes\n")

        return config_name, True, time_taken
    except FileNotFoundError:
        with open(log_file, "a") as f:
            f.write(f"Error: Script {command[1] if len(command) > 1 else 'UNKNOWN'} not found\n") # Use command[1] for script name
        return config_name, False, 0
    except subprocess.CalledProcessError as e:
        with open(log_file, "a") as f:
            f.write(f"Error running {model_type} model on {dataset_name} ({location}) data: {e}\n")
        return config_name, False, 0
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"An unexpected error occurred during {model_type}/{dataset_name}/{location}: {e}\n")
        return config_name, False, 0


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run HAR models in parallel")
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of parallel workers (default: 2)")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output (details will still be in log files)")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store log files (default: logs)")
    args = parser.parse_args()

    # Create a timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up log directory
    log_dir = args.log_dir
    if "{timestamp}" in log_dir:
        # Replace {timestamp} placeholder if present
        log_dir = log_dir.replace("{timestamp}", timestamp)

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    print(f"Starting HAR model comparison at {timestamp}")
    print(f"Using maximum {args.max_workers} parallel workers")
    print(f"Logs will be stored in: {os.path.abspath(log_dir)}")

    # Define datasets, models, and locations
    datasets = ["R1"]
    models = [
        "rf",
        "cnn",
        # "lstm",
        "transformer",
    ]
    locations_r1 = ["wrist", "thigh"]
    location_wisdm = "thigh"  # Treat WISDM phone-in-pocket as thigh location

    # Generate all configurations to run
    configurations = []
    for dataset in datasets:
        if dataset == "WISDM":
            # For WISDM, use the designated location
            for model in models:
                configurations.append((model, dataset, location_wisdm))
        elif dataset == "R1":
            # For R1, iterate through its specific locations
            for loc in locations_r1:
                for model in models:
                    configurations.append((model, dataset, loc))

    total_jobs = len(configurations)
    print(f"Total configurations to run: {total_jobs}")

    results = {}

    # Run configurations in parallel with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(run_model, model_type, dataset_name, location, log_dir): (
                model_type,
                dataset_name,
                location,
            )
            for model_type, dataset_name, location in configurations
        }

        # Create a progress bar
        completed = 0
        with tqdm(total=total_jobs, unit="job", desc="Progress", file=sys.stdout) as pbar:
            # Process results as they complete
            for future in as_completed(future_to_config):
                model_type, dataset_name, location = future_to_config[future]
                config_name, success, time_taken = future.result()
                results[config_name] = {"status": "Success" if success else "Failed", "time": time_taken}

                # Update progress
                completed += 1
                pbar.update(1)

                # Display compact information about completed job
                if not args.quiet:
                    status = "✓" if success else "✗"
                    print(
                        f"\r{status} {config_name} - {time_taken:.2f}min " + f"({completed}/{total_jobs})",
                        file=sys.stderr,
                    )

    # Print summary
    print("\n" + "=" * 80)
    print("HAR MODEL COMPARISON SUMMARY")
    print("=" * 80)

    # Group results by dataset and location for cleaner output
    by_dataset = {"WISDM": [], "R1_wrist": [], "R1_thigh": []}
    for config, result in results.items():
        model, dataset, location = config.split("_")
        key = dataset if dataset == "WISDM" else f"{dataset}_{location}"
        by_dataset[key].append((model, result))

    # Print grouped results
    for group_name, group_results in by_dataset.items():
        if group_results:
            print(f"\n{group_name.upper()}:")
            for model, result in sorted(group_results):
                status_str = f"{result['status']}"
                if result["status"] == "Success":
                    status_str += f" ({result['time']:.2f} min)"
                print(f"  {model.ljust(12)}: {status_str}")

    print("\n" + "=" * 80)
    print(f"Log files are available in: {os.path.abspath(log_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
