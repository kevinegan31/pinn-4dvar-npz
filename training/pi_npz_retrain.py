#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pytorch_lightning import seed_everything
from torch.utils.data import random_split, DataLoader, TensorDataset
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from sklearn.preprocessing import QuantileTransformer

import lightning as pl
import wandb
import time
import joblib
from joblib import Parallel, delayed

# Local imports ----------------------------------------------
# Add the directory containing model_utils.py to the Python path
import shutil
import os
import sys
# sys.path.append('/share/tempest2/egank31/pinn_model_utils/')
sys.path.append('/share/tempest2/egank31/pinn_model_utils/')
from pl_model_utils_frozen import DNN, PhysicsInformedNN

import torch
from torch import nn
import torch.distributed as dist
import numpy as np
import pandas as pd
from collections import OrderedDict
import datetime

from random import sample
import csv
import random
# Set a fixed seed for reproducibility
DATASET_IDX = os.getenv('DATASET_IDX', '01')
# SEED = 1000 * int(DATASET_IDX)
SEED = int(os.getenv('SEED', '1234'))
seed_everything(SEED, workers=True)
# seed_everything(1234, workers=True)  # Replace 1234 with your preferred seed value

# Initialize Distributed Processing Group
import torch.distributed as dist
from torch.distributed import init_process_group

# Wandb API key
os.environ["WANDB_API_KEY"] = "afbdb23b998f8a00b04076885fa26f4ec70f0a16"

# Ensure directory exists ----------------------------------------------
def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    
NUM_MINUTES = 10 #int(os.getenv('NUM_MINUTES', '15'))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dt_object = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
formatted_timestamp = dt_object.strftime("%Y_%m_%d")
NUM_LAYERS = int(os.getenv('NUM_LAYERS', '5'))
NUM_NEURONS = int(os.getenv('NUM_NEURONS', '96'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.005'))

def npz_nl(x, t, Vm, ks, m, Rm, ivlev, gamma, q):
    x = np.asarray(x)
    dx = np.zeros(x.shape)

    
    dx[1] = (((Vm * x[0]) / (ks + x[0])) * x[1]) - (m * x[1]) - (ivlev * x[1] * Rm * (1 - np.exp(-ivlev * x[1])) * x[2])
    dx[2] = ((1 - gamma) * ivlev * x[1] * Rm * (1 - np.exp(-ivlev * x[1])) * x[2]) - (q * x[2])
    dx[0] = -(((Vm * x[0]) / (ks + x[0])) * x[1]) + (m * x[1]) + (q * x[2]) + (gamma * ivlev * x[1] * Rm * (1 - np.exp(-ivlev * x[1])) * x[2])
    
    return dx
    
def rk4(f, x0, times, tfrc, frc, args):
    """
    Perform the model integration using the Runge-Kutta 4. This
    allows for the function to be forced at given times, ft, by the forcing, f.
    """
    nt = len(times)
    x = np.zeros((nt, len(x0)))
    x[0, :] = x0
    dt = np.zeros(nt)
    dt[1:] = np.diff(times)
    for n in range(1, nt):
        k1 = f(x[n - 1, :], times[n], *args) * dt[n]
        k2 = f(x[n - 1, :] + 0.5 * k1, times[n] + 0.5 * dt[n], *args) * dt[n]
        k3 = f(x[n - 1, :] + 0.5 * k2, times[n] + 0.5 * dt[n], *args) * dt[n]
        k4 = f(x[n - 1, :] + k3, times[n] + dt[n], *args) * dt[n]
        x[n, :] = x[n - 1, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if tfrc is not None:
            fl = np.where(np.logical_and(tfrc >= times[n] - 0.5 * dt[n],
                                         tfrc < times[n] + 0.5 * dt[n]))[0]
            x[n, :] += f[fl, :].sum(axis=0)

    return x

# RK4 for a single set of initial conditions
def run_rk4_for_initial_conditions(N0, P0, Z0, t, phi):
    # Define the initial conditions as a list
    x0 = [N0, P0, Z0]
    # Use the rk4 method to expand the system over time
    xt = rk4(npz_nl, x0, t, None, 0, phi)
    return xt

def forward_pinn(model, initial_state, nd_ntot, truth_t):
    model.eval()  # Just in case

    # Prepare initial state (normalize)
    initial_state_x0 = initial_state / nd_ntot
    forward_predictions = [initial_state_x0]  # Start with normalized x0

    for _ in range(len(truth_t) - 1):
        inp = forward_predictions[-1].unsqueeze(0)  # Shape [1, 3]
        with torch.no_grad():
            ut = model.net_u(inp)  # Shape [1, 3]
        forward_predictions.append(ut.squeeze(0))  # Shape [3]

    # Stack into tensor of shape [T, 3]
    forward_predictions_tensor = torch.stack(forward_predictions)

    # De-normalize output
    return forward_predictions_tensor * nd_ntot

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

# Read and preprocess the data --------------------------------------------------------------
# def read_and_preprocess_data(file_path, chunksize=10**6, device='cuda'):
#     print("Reading CSV file...")
#     sys.stdout.flush()
#     chunks = pd.read_csv(file_path, chunksize=chunksize)
#     df_list = [chunk for chunk in chunks]
#     df = pd.concat(df_list)
#     print("CSV file read successfully")
#     sys.stdout.flush()
#     dtype = torch.float32
#     X = df.iloc[:,1:-3].values
#     u = df.iloc[:, -3:].values
#     nd_ntot = np.ceil(X.sum(axis=1).max() * 100) / 100
#     X_modeling = X.copy()
#     u_modeling = u.copy()
#     X_modeling_nd = X_modeling.copy()
#     u_modeling_nd = u_modeling.copy()
#     # Non-dimensionalize N, P, Z. Keep time constant 1
#     X_modeling_nd = X_modeling / nd_ntot
#     u_modeling_nd = u_modeling / nd_ntot
#     X_modeling_nd = torch.tensor(X_modeling_nd, device=device, dtype=dtype)
#     u_modeling_nd = torch.tensor(u_modeling_nd, device=device, dtype=dtype)
#     return X_modeling_nd, u_modeling_nd, nd_ntot

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

# Environment variables
NEPOCH_ADAM = int(os.getenv('NEPOCH_ADAM', '1000')) #100 #int(os.getenv('NEPOCH_ADAM', '100'))
ACTIVATION_FUNCTION = os.getenv('ACTIVATION_FUNCTION', 'gelu')  # Keep as a string
CSV_PATH = os.getenv('CSV_PATH', './current_rk4_random_states_100000_df.csv')
stability_threshold = float(os.getenv('STABILITY_THRESHOLD', '0.9'))  # e.g., 0.95 for 95% skill
batch_size = int(os.getenv('BATCH_SIZE', '2048'))
# Map activation function names to PyTorch classes
activation_mapping = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh
}

# Validate activation function
if ACTIVATION_FUNCTION not in activation_mapping:
    raise ValueError(f"Unknown activation function: {ACTIVATION_FUNCTION}")

# Assign both a string (for logging) and a class (for model training)
ACTIVATION_FUNCTION_CLASS = activation_mapping[ACTIVATION_FUNCTION]  # Class for model

# Number of GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 1:
    raise ValueError("No GPUs available for training")

print(f"Number of GPUs: {num_gpus}")
sys.stdout.flush()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    # Checkpoint callback to save the best model

    # Read and preprocess the data
    print("Loading and Running Franks...")
    sys.stdout.flush()
    # Start with the rollout holdout data
    # rollout_df = pd.read_parquet("/share/tempest2/egank31/pinn_test_data/rollout_holdout_1500.parquet")
    rollout_df = pd.read_csv("/share/tempest2/egank31/pinn_test_data/npz_70far_30near_7_days_05min_val_holdout_set.csv")
    rollout_ics = rollout_df[['N', 'P', 'Z']].values
    # Constants
    Vm_franks, ks_franks, m_franks, gamma_franks, Rm_franks, ivlev_franks, q_franks = 2, 1, 0.1, 0.3, 1.5, 1, 0.2
    phi = (Vm_franks, ks_franks, m_franks, Rm_franks, ivlev_franks, gamma_franks, q_franks)

    # Time settings
    minutes_per_step_franks = 10
    dt_franks = minutes_per_step_franks / (60 * 24)  # Convert minutes to days
    tend_franks = 7 #10 / (60 * 24)  # 10-minute forecast horizon
    times_disc_franks = np.arange(0, tend_franks + dt_franks, dt_franks)[:-1]

    # Output container
    n_ics = rollout_ics.shape[0]
    n_steps = len(times_disc_franks)

    # Evolve each IC
    franks_trajectories = Parallel(n_jobs=20)(
        delayed(run_rk4_for_initial_conditions)(N0, P0, Z0, times_disc_franks, phi)
        for (N0, P0, Z0) in rollout_ics
    )

    print("Reading training data from CSV...")
    sys.stdout.flush()
    x, u, ND_NTOT = read_and_preprocess_data(CSV_PATH, chunksize=10**6, device='cpu')
    g = torch.Generator()
    g.manual_seed(1234)
    # Split dataset into training and validation sets
    # batch_size = 2048  # Fixed batch size for training
    full_dataset = TensorDataset(x, u)
    full_dataset_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        num_workers=8,  # Adjust based on your system
        pin_memory=True,  # Enable if using GPU
        # prefetch_factor=2,  # Optional: adjust based on your system
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        )
    # full_dataset_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    # Hyperparameters
    num_layers = NUM_LAYERS
    num_neurons = NUM_NEURONS
    learning_rate = LEARNING_RATE
    lambda_u = 1.0 #config["lambda_u"]
    lambda_f = 1.0 #config["lambda_f"]
    checkpoint_dir = f"/share/tempest2/egank31/checkpoints/retrain_frozen_params_{formatted_timestamp}_checkpoints_{NUM_MINUTES}_minutes_{num_layers}_layers_{num_neurons}_neurons_{learning_rate}_lr"

    # Check if the directory exists, and if not, create it
    ensure_dir(checkpoint_dir)
    # Parameters Dictionary
    dtype = torch.float32
    t_scale = 1.0
    alpha_tilde = 1.2164 * t_scale
    beta_tilde = 1.2795 * ND_NTOT
    b_tilde = 0.1 * t_scale
    c_tilde = 0.2 * t_scale
    e_tilde = 0.5 * t_scale * ND_NTOT
    f_tilde = 0.5 * t_scale * ND_NTOT
    params_dict = {
    'alpha_tilde': torch.tensor([alpha_tilde], dtype=dtype),
    'beta_tilde': torch.tensor([beta_tilde], dtype=dtype),
    'b_tilde': torch.tensor([b_tilde], dtype=dtype),
    'c_tilde': torch.tensor([c_tilde], dtype=dtype),
    'e_tilde': torch.tensor([e_tilde], dtype=dtype),
    'f_tilde': torch.tensor([f_tilde], dtype=dtype),
    }
    # Model
    model = PhysicsInformedNN(DNN, num_layers, num_neurons,
                              ACTIVATION_FUNCTION_CLASS, params_dict,
                              lambda_u, lambda_f, learning_rate, dtype=dtype)
    sys.stdout.flush()
    # Initialize WandbLogger
    n_obs = len(x)
    wandb_logger = WandbLogger(
        project=f"frozen_params_retrain_{formatted_timestamp}_{NUM_MINUTES}_minutes_{n_obs}_obs_{num_layers}_layers_{num_neurons}_neurons_{learning_rate}_lr",   # Replace with your WandB project name
        name=f"second_run_{NEPOCH_ADAM}_epochs_{formatted_timestamp}_dataset_{DATASET_IDX}",  # Unique name for each run using timestamp
        log_model=False,
        # save_dir="./checkpoints/"   # Optional local directory to save logs
    )
    hparams = {
        "Nt": ND_NTOT,
        "dataset_idx": DATASET_IDX,
        "activation_function": ACTIVATION_FUNCTION,
        "n_obs": n_obs,
        "lambda_u": lambda_u,
        "lambda_f": lambda_f,
        "learning_rate": learning_rate,
        "num_layers": num_layers,
        "num_neurons": num_neurons,
        "n_epochs": NEPOCH_ADAM,
        "seed": SEED,
        "batch_size": batch_size,
    }
    # checkpoint_name = f'optimal_{NEPOCH_ADAM}_total_epochs_retrain_opt_{DATASET_IDX}_dataset_{n_obs}_nobs_batch_size_{batch_size}_layers_{num_layers}_neurons_{num_neurons}_lr_{learning_rate}_activation_{ACTIVATION_FUNCTION}_{ND_NTOT}_Nt_{SEED}_seed_{NUM_MINUTES}_min_epoch{{epoch:04d}}'
    checkpoint_name = (
    f'optimal_{NEPOCH_ADAM}_epochs_retrain_opt_{DATASET_IDX}_dataset_'
    f'{n_obs}_nobs_batch_size_{batch_size}_layers_{num_layers}_neurons_{num_neurons}_'
    f'lr_{learning_rate}_activation_{ACTIVATION_FUNCTION}_{ND_NTOT}_Nt_{SEED}_seed_{NUM_MINUTES}_min'
    f'_{{epoch:04d}}'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        monitor=None,           # don’t monitor anything, just save periodically
        save_top_k=-1,          # keep all checkpoints
        every_n_epochs=500,    # or 500, depending on how fine you want
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')  # or 'epoch' if that fits your decay style
    # early_stopping = False
    trainer = pl.Trainer(max_epochs=NEPOCH_ADAM,
                         accelerator='gpu',
                         devices=num_gpus,
                         strategy='ddp',#if num_gpus > 1 else "auto",
                         deterministic=True,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, lr_monitor],
                         )
    print("Trainer setup complete. Starting the fit...")
    sys.stdout.flush()
    # Log hyperparameters only on rank-0 and when using DDP
    if trainer.is_global_zero:
        wandb_logger.experiment.config.update(hparams)
    # Start training
    start = time.time()
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, full_dataset_loader)
    end = time.time()
    survival_99_dict = {"frac_survived": -1.0}  # default value
    if trainer.is_global_zero:
        src = os.path.join(checkpoint_dir, "last.ckpt")
        dst = os.path.join(checkpoint_dir, f"{checkpoint_name}_final.ckpt")
        shutil.copy(src, dst)  # or shutil.move if you want to remove original
        os.remove(src)         # Delete the original last.ckpt to avoid clutter
        # =============================
        # Forward Rollout Evaluation
        # =============================
        print("Evaluating forward rollout stability and accuracy...")
        total_days = 7 # / 24
        intervals_per_day = 24 * 2 * 3 # 10 minute intervals
        # Create an array from 0 to 7 days with half-hour intervals
        times = np.arange(0, total_days + 1/intervals_per_day, 1/intervals_per_day)
        if times[-1] > total_days:
            times = times[:-1]
        # Model Evaluation
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rollout_tensor = torch.tensor(rollout_ics, dtype=torch.float32, device=model.device)

        def run_forward_pinn_single_safe(x0, model, nd_ntot, times):
            pred = forward_pinn(model, x0, nd_ntot, times).detach().cpu().numpy()
            if not np.isfinite(pred).all():
                return np.full((len(times), 3), np.nan)  # ensure correct shape
            return pred

        forward_predictions_list = Parallel(n_jobs=6)(
            delayed(run_forward_pinn_single_safe)(x0, model, ND_NTOT, times) for x0 in rollout_tensor
        )
        # Align exactly with all franks trajectories
        forward_actuals = [np.array(gt) for gt in franks_trajectories]  # No slicing!
        
        forward_preds = np.array(forward_predictions_list)  # Shape: [N, T, 3]
        forward_truths = np.array(forward_actuals)          # Shape: [N, T, 3]
        epsilon = 1e-8  # very small, avoids affecting non-zero values
        skill_scores = 1 - (np.abs(forward_preds - forward_truths) / (np.abs(forward_truths) + epsilon))

        # Now aggregate: mean and std over all trajectories (axis=0)
        # Resulting shapes: (num_time_steps, num_vars)
        # Set invalid entries (NaN or Inf) to a penalty value like -1
        # invalid_mask = ~np.isfinite(skill_scores)
        # skill_scores[invalid_mask] = -1.0
        # skill_mean = np.mean(skill_scores, axis=0)
        # survival_99 = compute_survival_metrics(skill_scores, threshold=0.99)
        # survival_score = survival_99["frac_survived"]
        # survival_99_dict = {'frac_survived': survival_score}
        wall_clock_hours = (end - start) / 3600
        gpu_hours = wall_clock_hours * num_gpus
        skill_scores_penalized = np.where(np.isfinite(skill_scores), skill_scores, -1.0)

        # Flatten-first (weights each timestep equally)
        mean_skill_flat = np.mean(skill_scores_penalized, axis=(0, 1))   # per-variable [3]
        global_skill_flat = np.mean(skill_scores_penalized)              # scalar
        # Trajectory-first (weights each trajectory equally)
        traj_mean_skill = np.mean(skill_scores_penalized, axis=1)        # [N, 3]
        mean_skill_traj = np.mean(traj_mean_skill, axis=0)               # per-variable [3]
        global_skill_traj = np.mean(traj_mean_skill)
        
        # Invalid fraction (for diagnostics)
        invalid_fraction = np.mean(~np.isfinite(skill_scores))
        # -----------------------------
        # Divergence & survival metrics
        # -----------------------------
        traj_len = forward_preds.shape[1]
        div_times = []
        for traj in np.mean(skill_scores_penalized, axis=2):  # average over vars per timestep
            diverged_idx = np.where(traj < stability_threshold)[0]
            div_times.append(diverged_idx[0] if len(diverged_idx) > 0 else traj_len)

        t_div = np.mean(div_times)
        avg_div_time = np.mean(t_div)
        median_div_time = np.median(t_div)

        total_rmse = calculate_global_rmse(forward_actuals, forward_predictions_list)

        # # Use penalized skill scores for survival metrics
        # survival_99 = compute_survival_metrics(skill_scores_penalized, threshold=0.90)
        # survival_99_dict = {"frac_survived": survival_99["frac_survived"]}
        
        # # objective_score = survival_99_dict['frac_survived']  * global_skill_flat
        # global_skill_clipped = np.clip(global_skill_flat, 0.0, 1.0)
        # # objective_score = survival_99_dict['frac_survived'] * global_skill_clipped
        # objective_score = (0.9 * survival_99_dict['frac_survived']) + (0.1 * global_skill_clipped)
        # Survival at 0.95
        stability = compute_survival_metrics(skill_scores_penalized, threshold=stability_threshold)
        frac_stable = stability["frac_survived"]

        # Clip skill to [0,1]
        global_skill_clipped = np.clip(global_skill_flat, 0.0, 1.0)

        # Penalty for invalid fraction
        validity_score = 1.0 - invalid_fraction  # higher is better

        # if invalid_fraction > 0.0:
        #     objective_score = 0.0
        # else:
        #     objective_score = (
        #         0.95 * frac_stable +
        #         0.05 * global_skill_clipped #+
        #         # 0.1 * validity_score
        #     )
        # -----------------------------
        # Log metrics
        # -----------------------------
        wandb_logger.log_metrics({
            "forward_total_rmse": total_rmse,
            "avg_divergence_time": avg_div_time,
            "median_divergence_time": median_div_time,
            # Flatten-first skills
            "mean_skill_N_flat": mean_skill_flat[0],
            "mean_skill_P_flat": mean_skill_flat[1],
            "mean_skill_Z_flat": mean_skill_flat[2],
            "global_skill_flat": global_skill_flat,
            # Trajectory-first skills
            "mean_skill_N_traj": mean_skill_traj[0],
            "mean_skill_P_traj": mean_skill_traj[1],
            "mean_skill_Z_traj": mean_skill_traj[2],
            "global_skill_traj": global_skill_traj,
            # Survival metrics
            f"tdiv_days_mean_{int(stability_threshold*100)}": stability["mean_days"],
            f"stability_rate_{int(stability_threshold*100)}": stability["frac_survived"],
            # Diagnostics
            "skill_invalid_fraction": invalid_fraction,
            # "Objective": objective_score
            "runtime_hours": wall_clock_hours,
            "gpu_hours": gpu_hours,
            "gpu_minutes": gpu_hours * 60,
            "gpu_seconds": gpu_hours * 3600
        })
        total_seconds = end - start
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60
        total_gpu_hours = total_hours * num_gpus
        total_gpu_minutes = total_gpu_hours * 60
        total_gpu_seconds = total_gpu_hours * 3600
        # Save to CSV
        summary_log_path = f"./runtime_logs/pinn_retrain_{SEED}_seed_{batch_size}_batches_{NEPOCH_ADAM}_epochs_{DATASET_IDX}_dataset_holdout_rollout_total_runtime_summary_{formatted_timestamp}.csv"
        os.makedirs("./runtime_logs", exist_ok=True)
        with open(summary_log_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "dataset_idx", "num_gpus",
                "total_seconds", "total_minutes", "total_hours",
                "total_gpu_hours", "total_gpu_minutes", "total_gpu_seconds",
                "num_epochs", "seed", "batch_size"
            ])
            writer.writeheader()
            writer.writerow({
                "timestamp": formatted_timestamp,
                "dataset_idx": DATASET_IDX,
                "num_gpus": num_gpus,
                "total_seconds": round(total_seconds, 1),
                "total_minutes": round(total_minutes, 1),
                "total_hours": round(total_hours, 2),
                "total_gpu_hours": round(total_gpu_hours, 2),
                "total_gpu_minutes": round(total_gpu_minutes, 1),
                "total_gpu_seconds": round(total_gpu_seconds, 0),
                "num_epochs": NEPOCH_ADAM,
                "seed": SEED,
                "batch_size": batch_size
            })
    # Sync survival_99 across all ranks (optional but recommended)
    if torch.distributed.is_initialized():
        obj_list = [survival_99_dict]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        survival_99_dict = obj_list[0]
    # Close the WandB run to allow new configurations
    if trainer.is_global_zero:
        wandb.finish()
    # Close the WandB run
