#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

import lightning as pl
import wandb
import time
from joblib import Parallel, delayed

# Local imports ----------------------------------------------
# Add the directory containing model_utils.py to the Python path
import shutil
import os
import sys
sys.path.append('./models/')
from pi_npz_model import DNN, PhysicsInformedNN
from traditional_npz import run_rk4_for_initial_conditions
sys.path.append('./utils/')
from additional_files import compute_survival_metrics, forward_pinn, read_and_preprocess_data, calculate_global_rmse

import torch
from torch import nn
import torch.distributed as dist
import numpy as np
import pandas as pd
from collections import OrderedDict
import datetime

import csv
import random
# Set a fixed seed for reproducibility
DATASET_IDX = os.getenv('DATASET_IDX', '01')
# SEED = 1000 * int(DATASET_IDX)
SEED = int(os.getenv('SEED', '1234'))
seed_everything(SEED, workers=True)

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
    # Read and preprocess the data
    print("Loading and Running Franks...")
    sys.stdout.flush()
    # Start with the rollout holdout data
    # rollout_df = pd.read_parquet("/share/tempest2/egank31/pinn_test_data/rollout_holdout_1500.parquet")
    rollout_df = pd.read_csv("./npz_70far_30near_7_days_holdout_set.csv")
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
    full_dataset = TensorDataset(x, u)
    full_dataset_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        num_workers=8,  # Adjust based on your system
        pin_memory=True,  # Enable if using GPU
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        )
    # Hyperparameters
    num_layers = NUM_LAYERS
    num_neurons = NUM_NEURONS
    learning_rate = LEARNING_RATE
    lambda_u = 1.0 #config["lambda_u"]
    lambda_f = 1.0 #config["lambda_f"]
    checkpoint_dir = f"./retrain_{formatted_timestamp}_checkpoints_{NUM_MINUTES}_minutes_{num_layers}_layers_{num_neurons}_neurons_{learning_rate}_lr"

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
        project=f"retrain_{formatted_timestamp}_{NUM_MINUTES}_minutes_{n_obs}_obs_{num_layers}_layers_{num_neurons}_neurons_{learning_rate}_lr",   # Replace with your WandB project name
        name=f"retrain_{NEPOCH_ADAM}_epochs_{formatted_timestamp}_dataset_{DATASET_IDX}",  # Unique name for each run using timestamp
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
    # checkpoint_name = (
    # f'optimal_{NEPOCH_ADAM}_epochs_retrain_opt_{DATASET_IDX}_dataset_'
    # f'{n_obs}_nobs_batch_size_{batch_size}_layers_{num_layers}_neurons_{num_neurons}_'
    # f'lr_{learning_rate}_activation_{ACTIVATION_FUNCTION}_{ND_NTOT}_Nt_{SEED}_seed_{NUM_MINUTES}_min'
    # f'_{{epoch:04d}}'
    # )
    checkpoint_name = 'pi_npz_final_{{epoch:04d}}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        monitor=None,           # don’t monitor anything, just save periodically
        save_top_k=-1,          # keep all checkpoints
        every_n_epochs=500,    # or 500, depending on how fine you want
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')  # or 'epoch' if that fits your decay style
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
        
        stability = compute_survival_metrics(skill_scores_penalized, threshold=stability_threshold)
        frac_stable = stability["frac_survived"]

        # Clip skill to [0,1]
        global_skill_clipped = np.clip(global_skill_flat, 0.0, 1.0)

        # Penalty for invalid fraction
        validity_score = 1.0 - invalid_fraction  # higher is better
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
