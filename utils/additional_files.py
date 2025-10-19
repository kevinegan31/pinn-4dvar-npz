import numpy as np
import torch
import pandas as pd

def compute_time_to_divergence(forward_actuals, forward_predictions,
                               rel_error_threshold=0.01, epsilon=1e-8):
    time_to_div = []

    for actual, pred in zip(forward_actuals, forward_predictions):
        # Compute per-timestep relative error across N, P, Z
        abs_diff = np.linalg.norm(pred - actual, axis=1)
        norm_truth = np.linalg.norm(actual, axis=1) + epsilon
        rel_error_t = abs_diff / norm_truth  # shape: (T,)

        diverged = np.where(rel_error_t > rel_error_threshold)[0]

        if len(diverged) == 0:
            time_to_div.append(len(pred))  # Fully stable
        else:
            time_to_div.append(diverged[0])  # First divergence point

    return np.array(time_to_div)

# RMSE function
def calculate_rmse(actual, predicted):
    residuals = actual - predicted
    rmse = np.sqrt(np.mean(residuals ** 2))
    return rmse

# Define a function to calculate relative error
def calculate_rel_error(actual, predicted):
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by zero warnings
        relative_error = np.abs(predicted - actual) / actual  # Relative Error
        relative_error = np.where(np.isnan(relative_error) | np.isinf(relative_error), 0, relative_error)  # Handle division by zero or NaN cases
    
    rel_error = np.mean(relative_error)
    return rel_error

def calculate_interval_rmse(actuals, predictions, total_intervals):
    rmse_N = []
    rmse_P = []
    rmse_Z = []
    std_error_N = []
    std_error_P = []
    std_error_Z = []
    for interval in range(total_intervals):
        actual_interval_N = [actual[interval, 0] for actual in actuals if interval < len(actual)]
        predicted_interval_N = [pred[interval, 0] for pred in predictions if interval < len(pred)]
        
        actual_interval_P = [actual[interval, 1] for actual in actuals if interval < len(actual)]
        predicted_interval_P = [pred[interval, 1] for pred in predictions if interval < len(pred)]
        
        actual_interval_Z = [actual[interval, 2] for actual in actuals if interval < len(actual)]
        predicted_interval_Z = [pred[interval, 2] for pred in predictions if interval < len(pred)]
        
        rmse_N.append(calculate_rmse(np.array(actual_interval_N), np.array(predicted_interval_N)))
        rmse_P.append(calculate_rmse(np.array(actual_interval_P), np.array(predicted_interval_P)))
        rmse_Z.append(calculate_rmse(np.array(actual_interval_Z), np.array(predicted_interval_Z)))
        
        std_error_N.append(np.std(rmse_N))  # Standard deviation of N errors
        std_error_P.append(np.std(rmse_P))  # Standard deviation of P errors
        std_error_Z.append(np.std(rmse_Z))  # Standard deviation of Z errors
        
    return (rmse_N, std_error_N), (rmse_P, std_error_P), (rmse_Z, std_error_Z)

def calculate_global_rmse(actuals, predictions):
    actual_all = np.vstack(actuals)      # Shape: [N*T, d]
    predicted_all = np.vstack(predictions)
    residuals = predicted_all - actual_all
    return np.sqrt(np.nanmean(residuals**2))  # Handles NaNs safely
    # return np.sqrt(np.nanmean(residuals**2)) if np.isfinite(residuals).all() else np.inf

def compute_survival_metrics(skill_scores, threshold, step_minutes=10):
    traj_len = skill_scores.shape[1]
    div_times = []
    for traj in np.nanmean(skill_scores, axis=2):  # Mean over vars
        diverged_idx = np.where(traj < threshold)[0]
        div_times.append(diverged_idx[0] if len(diverged_idx) > 0 else traj_len)

    div_times = np.array(div_times)
    t_div_days = div_times * (step_minutes / 60 / 24)
    full_window = traj_len * (step_minutes / 60 / 24)
    frac_survived = np.mean(div_times == traj_len)

    if frac_survived == 0.0:
        print(f"Warning: No trajectories survived the full window at threshold {threshold}")
        # Optionally clamp or log this case depending on use:
        # return a penalty or NaN-safe value if needed for HPO
        # e.g., return very low value to penalize this config:
        return {
            "mean_days": np.mean(t_div_days),
            "median_days": np.median(t_div_days),
            "frac_survived": 0.0
        }

    return {
        "mean_days": np.mean(t_div_days),
        "median_days": np.median(t_div_days),
        "frac_survived": frac_survived
    }

def read_and_preprocess_data(file_path, chunksize=10**6, device='cuda'):
    print("Reading CSV file...")
    chunks = pd.read_csv(file_path, chunksize=chunksize)
    df = pd.concat(chunks)
    print("CSV file read successfully")

    dtype = torch.float32

    # Inputs: N0, P0, Z0
    X = df.iloc[:, :3].values
    # Targets: N10, P10, Z10
    u = df.iloc[:, 3:].values

    # Non-dimensionalize by max Ntot
    nd_ntot = np.ceil(max(X.sum(axis=1).max(), u.sum(axis=1).max()) * 100) / 100
    X_nd = X / nd_ntot
    u_nd = u / nd_ntot

    # Convert to tensors
    X_nd = torch.tensor(X_nd, device=device, dtype=dtype)
    u_nd = torch.tensor(u_nd, device=device, dtype=dtype)

    return X_nd, u_nd, nd_ntot