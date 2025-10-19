#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import csv
import datetime

import numpy as np
from scipy.integrate import odeint
from scipy.sparse.linalg import LinearOperator, cg

from pytorch_lightning import seed_everything


import importlib
import subprocess
import sys
import joblib

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"{package} not found, installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package)

# Standard library imports
import os
import time
import random
import copy
import multiprocessing
import itertools
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import json
import re
# Third-party imports
third_party_packages = ['numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'torch']

for package in third_party_packages:
    install_and_import(package)

# Specific imports from installed packages
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import torch.multiprocessing as mp

from pytorch_lightning import seed_everything
from torch.utils.data import random_split, DataLoader, TensorDataset
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from joblib import Parallel, delayed
from copy import deepcopy
# Local imports
# Add the directory containing model_utils.py to the Python path
sys.path.append('../models/')
from pi_npz import DNN as pi_npz_DNN
from pi_npz import PhysicsInformedNN
from pi_npz import forward_pinn_assimilation as forward_pinn, compute_jacobians, propagate_tlm, propagate_adjoint, make_innerloop_pinn
from traditional_npz import npz_nl, rk4, run_rk4_for_initial_conditions, npz_tl, rk4_ad


def make_innerloop(xb_rk4, timesteps, background_error_rk4):
    def innerloop(w):
        # print("\n--- INNERLOOP ---")
        # print("Input w:", w[:5])
        # Step 1: Construct full forcing array for adjoint model
        frc_ad = np.zeros((len(timesteps), 3))  # Full time grid
        frc_ad[obs_idx, obs_type] = w         # Inject w into observed components
        tfrc_ad = timesteps
        # print("Adjoint forcing (first 3):", frc_ad[:3])
        
        # Step 2: Run adjoint model
        ad_x0 = np.zeros(3)
        ad = rk4_ad(npz_nl, npz_ad, xb_rk4, ad_x0, timesteps, tfrc_ad, frc_ad, phi)
        # print("Adjoint final state:", ad[-1])
    
        # Step 3: Apply background covariance
        d = background_error_rk4.dot(ad.T).T

        # print("After background covariance (first state):", d[0])
    
        # Step 4: Construct full forcing array for tangent-linear model
        frc_tl = np.zeros((len(timesteps), 3))
        frc_tl[obs_idx, obs_type] = d[obs_idx, obs_type]
        tfrc_tl = timesteps
    
        # Step 5: Run tangent-linear model
        d_tl = rk4_tl(npz_nl, npz_tl, xb_rk4, d[0, :], timesteps, tfrc_tl, frc_tl, phi)
        # print("TLM final state:", d_tl[-1])
    
        # Step 6: Final projection and return
        result = d_tl[obs_idx, obs_type] + obs_error * w
        # print("Output (first 5):", result[:5])
        
        return result
    return innerloop


def make_innerloop_pinn(precomputed_jacobians, state_matrix_nd_tensor):
    def innerloop_pinn(w):
        num_states = state_matrix_nd_tensor.shape[0]
        num_features_tlm = state_matrix_nd_tensor.shape[1] + 1
        num_features_adj = state_matrix_nd_tensor.shape[1]
        
        # --- Step 1: adjoint forcing (3D) ---
        _, frc_ad_np = obs_forcing(obs_time, obs_type, w, truth_t)   # returns forcing in state-space
        frc_ad_np = np.array(frc_ad_np, dtype=float)  # ensure numpy
        # shape: (num_states, 3)

        # --- Step 2: adjoint propagation (strip bias from Jacobians) ---
        predicted_ad_forward = propagate_adjoint(
            precomputed_jacobians=[J[:3, :3] for J in precomputed_jacobians],  # 3x3 Jacobians
            frc_ad_np=frc_ad_np,     # forcing already 3D
            num_states=num_states,
            num_features=num_features_adj,          # adjoint dimension = 3
            dtype=dtype,
            device=device,
            lambda_T=None            # zero-terminal adjoint here
        )

        # In innerloop
        bg_error_tensor = torch.tensor(B0, dtype=dtype, device=device)# shape (3,3)
        d = predicted_ad_forward @ bg_error_tensor.T   # (num_states, 3)

        # --- Step 4: TLM forcing (4D: state + bias) ---
        d_obs = d[obs_idx, obs_type].detach().cpu().numpy()
        _, frc_tl_np = obs_forcing(obs_time, obs_type, d_obs, truth_t)
        frc_tl_np = np.concatenate([frc_tl_np, np.zeros((len(frc_tl_np), 1))], axis=1)
        forcing_matrix_tlm = torch.tensor(frc_tl_np, dtype=dtype, device=device)

        # --- Step 5: TLM propagation (4D) ---
        tl_x0 = d[0, :3].detach().cpu().numpy()   # initial perturbation (3D)
        # num_features = state_matrix_nd_tensor.shape[1] + 1   # 3 states + bias = 4
        predicted_tlms = propagate_tlm(
            precomputed_jacobians, tl_x0, forcing_matrix_tlm,
            num_states, num_features_tlm, dtype, device
        )

        # --- Step 6: final projection ---
        return predicted_tlms[obs_idx, obs_type] + obs_error * w

    return innerloop_pinn

def obs_forcing(otime, otype, frc_vals, time_step):
    """
    Return time-aligned forcing array of shape (nt, 3)
    for use in RK4 adjoint or tangent-linear integration.
    """
    otime = np.atleast_1d(otime)
    otype = np.atleast_1d(otype)
    frc_vals = np.atleast_1d(frc_vals)

    frc_aligned = np.zeros((len(time_step), 3))  # same time dimension as model state
    time_index = np.searchsorted(time_step, otime)  # map obs times to time grid indices

    for i, idx in enumerate(time_index):
        if 0 <= idx < len(time_step):
            frc_aligned[idx, otype[i]] += frc_vals[i]

    return time_step, frc_aligned

# Set the seed for reproducibility
np.random.seed(42)
### Load in Network ----------------------
# CPU device
dtype = torch.float64
device = torch.device('cpu') # Smaller models/data, running on CPU

# Load the checkpoint file path
# gelu_checkpoint_path = "../../checkpoints/evolved_states_optuna_trial_0_ntot_500000_nobs_batch_size_64_layers_4_neurons_128_lr_0.001_activation_gelu_lu_1.0_lf_1.0_20_patience_2.9_Nt_10_min.ckpt"
pi_npz_model_ckpt_name = 'pi_npz_final.ckpt'
pi_npz_checkpoint_path = f"../model_checkpoints/{pi_npz_model_ckpt_name}"

t_scale = 1.0
nd_ntot = 2.75
alpha_tilde = 1.2164 * t_scale
beta_tilde = 1.2795 * nd_ntot
b_tilde = 0.1 * t_scale
c_tilde = 0.2 * t_scale
e_tilde = 0.5 * t_scale * nd_ntot
f_tilde = 0.5 * t_scale * nd_ntot
# Define parameters for GELU model
params_dict = {
'alpha_tilde': torch.tensor([alpha_tilde], dtype=dtype),
'beta_tilde': torch.tensor([beta_tilde], dtype=dtype),
'b_tilde': torch.tensor([b_tilde], dtype=dtype),
'c_tilde': torch.tensor([c_tilde], dtype=dtype),
'e_tilde': torch.tensor([e_tilde], dtype=dtype),
'f_tilde': torch.tensor([f_tilde], dtype=dtype),
}
print("NUM_LAYERS =", os.getenv("NUM_LAYERS"))
print("NUM_NEURONS =", os.getenv("NUM_NEURONS"))
NUM_LAYERS = int(os.getenv('NUM_LAYERS', '5'))
NUM_NEURONS = int(os.getenv('NUM_NEURONS', '384'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.006'))
learning_rate_gelu = LEARNING_RATE
num_layers_gelu = NUM_LAYERS
num_neurons_gelu = NUM_NEURONS
activation_function = nn.GELU

# Step 1: Load the checkpoint manually
pi_npz_checkpoint = torch.load(pi_npz_checkpoint_path, map_location="cpu")

# Step 2: Reconstruct the model using the same init args
pi_npz_model = PhysicsInformedNN(
    DNN=pi_npz_DNN,
    num_layers=num_layers_gelu,
    num_neurons=num_neurons_gelu,
    activation_function=activation_function,
    params_dict=params_dict,
    lambda_u=1,
    lambda_f=1,
    learning_rate=learning_rate_gelu,
    dtype=dtype
)

# Step 3: Load the model weights from the checkpoint
pi_npz_model.load_state_dict(pi_npz_checkpoint["state_dict"])

# Step 4: Optional – move to float64
pi_npz_model = pi_npz_model.to(dtype=dtype, device="cpu")
print("Models loaded.")
sys.stdout.flush()
# Time Parameters
total_days = 7 #23.5 / 24

intervals_per_day = 24 * 2 * 3 # Half Hour intervals (2 per hour)

# Calculate the total number of half-hour intervals
total_intervals = total_days * intervals_per_day

# Create an array from 0 to 7 days with half-hour intervals
times = np.arange(0, total_days + 1/intervals_per_day, 1/intervals_per_day)

if times[-1] > total_days:
    times = times[:-1]

# Initial Franks RK4 for obs data
x0 = [1.6, 0.3, 0.1]
Vm = 2
ks = 1
m = 0.1
gamma = 0.3
Rm = 1.5
ivlev = 1
q = 0.2

tstart = 0 # Start time (day 0)
tend = total_days # End time (day 8)
minutes_per_step = 10 # Time step in minutes
minutes_per_hour = 60 # Constant time per hour
hours_per_day = 24 # Constant time per day
dt = minutes_per_step / (minutes_per_hour * hours_per_day)
t = np.arange(tstart, tend + dt, dt)  # Create the array of times

phi = (Vm, ks, m, Rm, ivlev, gamma, q)

xt = rk4(npz_nl, x0, t, None, 0, phi)

# Based on background_error_generation file
# Background Uncertainty N: 0.006207044131222433, Background Uncertainty P: 0.0038255009334436734, Background Uncertainty Z: 0.0021741729567239638
# Perturbation N: 0.07878479632024464, Perturbation P: 0.061850634058541985, Perturbation Z: 0.04662802758774988 -- np.sqrt(N + threshold_N)
xb_error_N, xb_error_P, xb_error_Z = 0.006, 0.004, 0.002

# Number of perturbations to possibly coose from
n_samples = 100000

# Standard deviations
sigma_N = 0.07878479632024464
sigma_P = 0.061850634058541985
sigma_Z = 0.04662802758774988

# Generate noise arrays
perturbation_N = np.random.normal(loc=0, scale=sigma_N, size=n_samples)
perturbation_P = np.random.normal(loc=0, scale=sigma_P, size=n_samples)
perturbation_Z = np.random.normal(loc=0, scale=sigma_Z, size=n_samples)

# Parameters
max_clipped_z = 500  # Allow at most 500 clipped Z values
# Number of background samples to generate
NUM_XB0 = int(os.getenv('NUM_XB0', '10000'))
bg_sample_indices = np.random.choice(len(perturbation_N), size=NUM_XB0, replace=False)
# Dictionary to store results
xb_dict = {}

# # Define true initial state
n0_true, p0_true, z0_true = xt[0,:]
MIN_GUESS_VAL = float(os.getenv('MIN_GUESS_VAL', '0.001'))
print("min guess val =", MIN_GUESS_VAL)

# Generate initial noise samples (oversample to allow for rejection)
oversample_factor = 1.5
sample_pool_size = int(NUM_XB0 * oversample_factor)
z0_samples = z0_true + np.random.normal(loc=0, scale=sigma_Z, size=sample_pool_size)

# Identify clipped values
clipped_mask = z0_samples < MIN_GUESS_VAL
z0_samples[clipped_mask] = MIN_GUESS_VAL

# Count how many were clipped
num_clipped = np.sum(clipped_mask)

# If too many values were clipped, redraw until we meet the limit
while num_clipped > max_clipped_z:
    excess = num_clipped - max_clipped_z
    # Resample only the clipped ones that exceed the cap
    redraw_indices = np.where(clipped_mask)[0][:excess]
    new_samples = z0_true + np.random.normal(loc=0, scale=sigma_Z, size=excess)
    z0_samples[redraw_indices] = new_samples

    # Recalculate clipping
    clipped_mask = z0_samples < MIN_GUESS_VAL
    z0_samples[clipped_mask] = MIN_GUESS_VAL
    num_clipped = np.sum(clipped_mask)

# Now trim to desired number of samples
initial_guesses_z = z0_samples[:NUM_XB0]
initial_guesses_n = np.clip(n0_true + perturbation_N[bg_sample_indices], MIN_GUESS_VAL, None)
initial_guesses_p = np.clip(p0_true + perturbation_P[bg_sample_indices], MIN_GUESS_VAL, None)
initial_guesses = list(zip(initial_guesses_n, initial_guesses_p, initial_guesses_z))

# Create obs data -----------------------------------------
nitrate_std_mg = 0.002 * 14.0067 # High-Sensitivity Nitrate plus Nitrite by Chemiluminescence
sigma_N = nitrate_std_mg # https://hahana.soest.hawaii.edu/hot/protocols/protocols.html# nitrate + nitrate
# Estimate Prochlorococcus nitrogen biomass and measurement uncertainty
# Assume Redfield ratio: C:N = 6.6 -> N content ~ 0.0000076 micrograms N per cell
# Typical cell abundance: ~10**5 cells/mL = 10**8 cells/L
# Prochlorococcus biomass: ~50 fg C per cell = 0.00005 micrograms C
# Nitrogen biomass = 10**8 cells/L * 0.0000076 micrograms N per cell = 0.76 micrograms N per liter
# 5% precision -> uncertainty:
sigma_P = 0.05 * 0.76 # Bacteria and Cyanobacteria by Flow Cytometry from https://hahana.soest.hawaii.edu/hot/protocols/protocols.html# and HOTDOGS bottle extraction https://hahana.soest.hawaii.edu/hot/protocols/protocols.html#
# Based on ~1–2% instrument precision for carbon concentration measurements (Maas et al. 2021)
# Allometry and the calculation of zooplankton metabolism in the subarctic Northeast Pacific Ocean paper 
# https://www.pnas.org/doi/10.1073/pnas.2404460121 
# For typical values of Z around 0.05 microgram N/L, this corresponds to ~20% relative uncertainty
# Conservative estimate used for measurement noise in assimilation
sigma_Z = 0.01  # microgram N/L; represents measurement precision, not total uncertainty
measurement_uncertainty = np.array([sigma_N, sigma_P, sigma_Z])

# error = np.array([1e-10, 1e-10, 1e-10])
# --- Define time window ---
period_end = 7 # Get exactly 7 days
period = (0, period_end)
rng = np.where((t >= period[0]) & (t <= period[1]))[0]
truth_t = t[rng]
truth = xt[rng, :]

n_days = 7
obs_per_day = 2
total_obs_per_species = n_days * obs_per_day  # → 14

# Get all time indices
all_indices = np.arange(len(truth))
time_days = truth_t

# Shared observation times for all species
shared_obs_times = []

for day in range(n_days):
    # Get indices for current day
    day_mask = (time_days >= day) & (time_days < day + 1)
    day_indices = all_indices[day_mask]

    # Randomly pick 2 unique time indices per day
    sampled_indices = np.random.choice(day_indices, size=obs_per_day, replace=False)
    shared_obs_times.extend(sampled_indices)

# Sort for consistency
shared_obs_times = np.sort(shared_obs_times)

# Now use same times for N, P, Z
obs_times_N = shared_obs_times.copy()
obs_times_P = shared_obs_times.copy()
obs_times_Z = shared_obs_times.copy()

# Confirm
print(f"Shared obs/day: {len(shared_obs_times) / n_days}")

# --- Combine indices and field labels ---
pts = np.concatenate([obs_times_N, obs_times_P, obs_times_Z])
fld = np.concatenate([
    np.zeros_like(obs_times_N),   # 0 = N
    np.ones_like(obs_times_P),    # 1 = P
    np.full_like(obs_times_Z, 2)  # 2 = Z
]).astype(int)

# --- Filter in-bounds ---
valid = pts < len(truth)
pts = pts[valid]
fld = fld[valid]
nobs = len(pts)  # Update number of observations
rng = np.arange(nobs)  # Update the range index

# --- Generate noisy observations ---
obs_idx = pts
obs_type = fld
obs_time = truth_t[obs_idx]
obs_plot = truth[obs_idx, :]
obs_error = measurement_uncertainty[obs_type] ** 2
# obs_value = obs_plot[np.arange(len(obs_idx)), obs_type] + measurement_uncertainty[obs_type]# * np.random.randn(len(obs_idx))
obs_value = obs_plot[np.arange(len(pts)), fld] + measurement_uncertainty[fld] * np.random.randn(len(pts))

# --- Remove invalid observations (e.g., negative concentrations) ---
valid_mask = obs_value > 0
obs_idx = obs_idx[valid_mask]
obs_type = obs_type[valid_mask]
obs_time = obs_time[valid_mask]
obs_error = obs_error[valid_mask]
obs_value = obs_value[valid_mask]
obs_plot = obs_plot[valid_mask]
fld = fld[valid_mask]
pts = pts[valid_mask]           # <- Needed
nobs = len(pts)  # Update number of observations
rng = np.arange(nobs)  # Update the range index

# --- Construct sparse observation matrix ---
obs_plot_new = np.zeros((len(obs_time), 3))
obs_plot_new[np.arange(len(obs_time)), obs_type] = obs_value
### 1. Run RK4 in parallel across CPU cores
def rk4_worker(i, n0, p0, z0):
    xb_0 = np.array([n0, p0, z0])
    start = time.time()
    xb_rk4 = rk4(npz_nl, xb_0, truth_t, None, 0, phi)
    end = time.time()
    duration = end - start
    return i, xb_0, xb_rk4, duration

num_jobs = int(os.getenv('NUM_JOBS', '4'))
start_time = time.time()
rk4_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=10)(
    delayed(rk4_worker)(i, n, p, z) for i, (n, p, z) in enumerate(initial_guesses)
)
end_time = time.time()
rk4_loop_time = end_time - start_time
print(f"Running time: {rk4_loop_time:.2f}")
# Print the number of trajectories computed
print(f"RK4 Complete - {len(rk4_results)} trajectories computed.")
# Store intermediate RK4 results for GPU PINN stage
rk4_dict = {i: {"perturbed_initial_state": xb_0,
                "background_state": xb_rk4,
                "run_time": duration}
            for i, xb_0, xb_rk4, duration in rk4_results}

# === Worker Function ===
def pinn_worker(i, x0_np, nd_ntot, truth_t):
    global pi_npz_model  # use global model, don’t pickle it into each worker
    device = next(pi_npz_model.parameters()).device

    x0_tensor = torch.tensor(x0_np / nd_ntot, dtype=dtype, device=device)

    start = time.time()
    pred_tensor, _ = forward_pinn(pi_npz_model, x0_tensor, truth_t)
    pred = pred_tensor.detach().cpu().numpy() * nd_ntot
    end = time.time()

    if i % 1000 == 0:
        print(f"Completed worker {i}")
        sys.stdout.flush()

    return i, x0_np, pred, end - start


# Convert entire list to array once (faster indexing, avoid zip unpacking)
initial_guesses_array = np.array(initial_guesses)  # shape (B, 3)
start_time = time.time()
print(f"Start time: {datetime.datetime.fromtimestamp(start_time)}")
# Spawn processes with model initialized globally
pi_npz_xb0_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=100)(
    delayed(pinn_worker)(i, x0_np, nd_ntot, truth_t)
    for i, x0_np in enumerate(initial_guesses_array)
)
end_time = time.time()
pinn_prediction_loop_time = end_time - start_time
print(f"Running time: {pinn_prediction_loop_time:.2f}")
print(f"PINN Forward Prediction Complete - {len(pi_npz_xb0_results)} trajectories computed.")
# === Rebuild Dictionary ===
### PI-NPZ Background Results
pi_npz_xb0_dict = {
    i: {
        "perturbed_initial_state": xb_0,
        "background_state": pred,
        "run_time": duration
        
    }
    for i, xb_0, pred, duration in pi_npz_xb0_results
}
# === Merge RK4 and PINN results into one dictionary ===
xb_dict = {}

for i in rk4_dict:
    xb_dict[i] = {
        "truth": truth,
        "perturbed_initial_state": rk4_dict[i]["perturbed_initial_state"],
        "rk4_background_state": rk4_dict[i]["background_state"],
        "pi_npz_background_state": pi_npz_xb0_dict[i]["background_state"],
        # "franks_pinn_background_state": franks_pinn_xb0_dict[i]["background_state"],
        # "reg_nn_background_state": reg_nn_xb0_dict[i]["background_state"],
        "rk4_run_time": rk4_dict[i]["run_time"],
        "pi_npz_run_time": pi_npz_xb0_dict[i]["run_time"],
        # "franks_pinn_run_time": franks_pinn_xb0_dict[i]["run_time"],
        # "reg_nn_run_time": reg_nn_xb0_dict[i]["run_time"],
    }


# === Save the combined dictionary ===
joblib.dump(xb_dict, f"/share/tempest2/egank31/assimilation_results/two_samples_per_day_xb_data_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_frozen_params_{NUM_LAYERS}_{NUM_NEURONS}_compressed.pkl")
print(f"Xb Data saved - {len(xb_dict)} trajectories computed.")
# Define background error model
background_error = np.array([xb_error_N, xb_error_P, xb_error_Z])
# Background error covariance
B0 = np.diag(background_error**2)

num_cg_iterations = int(os.getenv('NUM_CG', '2000'))
def safe_pct_drop(before, after):
    if before == 0:
        return np.nan  # or 0.0 if you prefer
    return ((before - after) / before)


def run_rk4_assimilation_for_trajectory(
    key, xb_data, background_error_rk4, obs_idx, obs_type, obs_value,
    obs_error, nobs, num_cg_iterations, truth_t, npz_nl, npz_ad, phi
):
    # Print progress every 1000 samples (rank-independent)
    if key % 1000 == 0:
        print(f"Completed assimilation for {key} trajectories.")
        
    xb_0_rk4 = xb_data["perturbed_initial_state"]
    xb_rk4 = xb_data["rk4_background_state"]
    truth = xb_data["truth"]
    
    # For RK4
    jo_b_rk4 = np.sum(((xb_rk4[obs_idx, obs_type] - obs_value) ** 2) / obs_error)
    jb_b_rk4 = np.sum(((xb_rk4[0, :] - xb_0_rk4) ** 2) / np.diag(background_error_rk4))
    J_total_b_rk4 = 0.5 * jb_b_rk4 + 0.5 * jo_b_rk4

    b = obs_value - xb_rk4[obs_idx, obs_type]
    cg_iter_count = [0]

    def callback(xk):
        cg_iter_count[0] += 1

    innerloop = make_innerloop(xb_rk4, truth_t, background_error_rk4)
    A = LinearOperator((nobs, nobs), matvec=innerloop)

    start_time = time.time()
    x, exit_code = cg(A, b, rtol=1e-13, maxiter=num_cg_iterations, callback=callback)
    end_time = time.time()
    total_time = end_time - start_time
    converged = int(exit_code == 0)

    tfrc, frc = obs_forcing(obs_time, obs_type, x, truth_t)
    ad_x0 = np.zeros(3)
    ad = rk4_ad(npz_nl, npz_ad, xb_rk4, ad_x0, truth_t, tfrc, frc, phi)
    z = background_error_rk4.dot(ad.T).T[0]

    xa_rk4 = rk4(npz_nl, xb_0_rk4 + z, truth_t, None, 0, phi)

    jo_a_rk4 = np.sum(((xa_rk4[obs_idx, obs_type] - obs_value) ** 2) / obs_error)
    jb_a_rk4 = np.sum(((xa_rk4[0, :] - xb_0_rk4) ** 2) / np.diag(background_error_rk4))
    J_total_a_rk4 = 0.5 * jb_a_rk4 + 0.5 * jo_a_rk4

    misfitb_rk4 = np.sqrt(np.sum((truth - xb_rk4) ** 2))
    misfita_rk4 = np.sqrt(np.sum((truth - xa_rk4) ** 2))
    improvement_rk4 = 100 * ((misfitb_rk4 - misfita_rk4) / misfitb_rk4)
    
    # pct_drop_J = ((J_total_b_rk4 - J_total_a_rk4) / J_total_b_rk4)
    # pct_drop_Jo = ((jo_b_rk4 - jo_a_rk4) / jo_b_rk4)
    # pct_drop_Jb = ((jb_b_rk4 - jb_a_rk4) / jb_b_rk4)
    pct_drop_J  = safe_pct_drop(J_total_b_rk4, J_total_a_rk4)
    pct_drop_Jo = safe_pct_drop(jo_b_rk4, jo_a_rk4)
    pct_drop_Jb = safe_pct_drop(jb_b_rk4, jb_a_rk4)

    result_dict = {
        "xa_rk4": xa_rk4,
        "xb_rk4": xb_rk4,
        "jo_b_rk4": jo_b_rk4,
        "jb_b_rk4": jb_b_rk4,
        "J_total_b_rk4": J_total_b_rk4,
        "jo_a_rk4": jo_a_rk4,
        "jb_a_rk4": jb_a_rk4,
        "J_total_a_rk4": J_total_a_rk4,
        "rk4_misfitb": misfitb_rk4,
        "rk4_misfita": misfita_rk4,
        "Improvement_rk4": improvement_rk4,
        "pct_drop_J_rk4": pct_drop_J,
        "pct_drop_Jo_rk4": pct_drop_Jo,
        "pct_drop_Jb_rk4": pct_drop_Jb,
        "cg_iterations_rk4": cg_iter_count[0],
        "converged_rk4": converged,
        "exit_code_rk4": exit_code,
        "rk4_time": total_time,
    }

    return key, result_dict

# Pack shared arguments
shared_args = {
    "background_error_rk4": B0,
    "obs_idx": obs_idx,
    "obs_type": obs_type,
    "obs_value": obs_value,
    "obs_error": obs_error,
    "nobs": nobs,
    "num_cg_iterations": num_cg_iterations,
    "truth_t": truth_t,
    "npz_nl": npz_nl,
    "npz_ad": npz_ad,
    "phi": phi,
}

# Run assimilation in parallel
rk4_assimilation_results_parallel = Parallel(n_jobs=num_jobs, backend="loky", batch_size=10)(
    delayed(run_rk4_assimilation_for_trajectory)(key, xb_data, **shared_args)
    for key, xb_data in xb_dict.items()
)

# Reconstruct result dictionary
rk4_assimilation_results = {key: result for key, result in rk4_assimilation_results_parallel}
print(f"RK4 Assimilation complete - {len(rk4_assimilation_results)} trajectories computed.")

joblib.dump(rk4_assimilation_results, f"/share/tempest2/egank31/assimilation_results/two_samples_per_day_rk4_assimilation_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_{num_cg_iterations}_compressed.pkl")
### PINN Jacobian Calculation
# Prepare shared arguments
pi_npz_shared_args = dict(
    model=pi_npz_model,
    dtype=dtype,
    nd_ntot=nd_ntot,
)

# Run in parallel
pi_npz_jacobian_start_time = time.time()
pi_npz_jacobian_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=10)(
    delayed(compute_jacobians)(
        model=pi_npz_shared_args["model"],                        # <-- pass model explicitly
        nd_trajectory=xb_data["pi_npz_background_state"] / nd_ntot,  # rescale to ND
        nd_ntot=nd_ntot,
        return_dimensional=True,                                  # get dimensional Jacobians
        dtype=dtype,
        device=device
    )
    for key, xb_data in xb_dict.items()
)
pi_npz_jacobian_end_time = time.time()

# Convert results back to dictionary (preserve key association)
pi_npz_jacobians_per_trajectory = {
    key: jacobians
    for (key, _), jacobians in zip(xb_dict.items(), pi_npz_jacobian_results)
}

print(f"Jacobian Matrix Run time for all matrices (parallel): "
      f"{pi_npz_jacobian_end_time - pi_npz_jacobian_start_time:.2f}s")
sys.stdout.flush()

# Rebuild dict
pi_npz_assimilation_results = {key: result for key, result in pi_npz_assimilation_results_parallel}
print(f"PI-NPZ Assimilation complete - {len(pi_npz_assimilation_results)} trajectories computed.")
joblib.dump(pi_npz_assimilation_results, f"../data/assimilation_results/two_samples_per_day_pi_npz_frozen_params_assimilation_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_{num_cg_iterations}_{NUM_LAYERS}_{NUM_NEURONS}_compressed.pkl")
