#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import warnings
import joblib

import numpy as np
import torch

from torch import nn
from scipy.sparse.linalg import LinearOperator, cg
from joblib import Parallel, delayed

# Repository paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_DIR = os.path.join(REPO_ROOT, "models")
OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "data",
    "nn_assimilation_results"
)

sys.path.append(MODEL_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

from nn_npz import (
    DNN as nn_npz_DNN,
    NN,
    forward_nn_assimilation as forward_nn,
    compute_jacobians,
    propagate_tlm,
    propagate_adjoint,
)

from traditional_npz import npz_nl, rk4

def make_innerloop_nn(precomputed_jacobians, state_matrix_nd_tensor):
    def innerloop_nn(w):
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

    return innerloop_nn

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

warnings.filterwarnings('ignore')

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Set the seed for reproducibility
np.random.seed(42)
### Load in Network ----------------------
# num_threads = torch.get_num_threads()
# torch.set_num_threads(num_threads)

# CPU device
dtype = torch.float64
device = torch.device('cpu') # Smaller models/data, running on CPU
nd_ntot = 2.75

# Load the checkpoint file path
nn_npz_checkpoint_path = os.path.join(
    REPO_ROOT,
    "model_checkpoints",
    "nn_npz_final.ckpt",
)

print("NUM_LAYERS =", os.getenv("NUM_LAYERS"))
NUM_LAYERS = int(os.getenv('NUM_LAYERS', '5'))
NUM_NEURONS = int(os.getenv('NUM_NEURONS', '512'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0001'))
learning_rate_gelu = LEARNING_RATE
num_layers_gelu = NUM_LAYERS
num_neurons_gelu = NUM_NEURONS
activation_function = nn.GELU

# Step 1: Load the checkpoint manually
nn_npz_checkpoint = torch.load(nn_npz_checkpoint_path, map_location="cpu")

# Step 2: Reconstruct the model using the same init args
nn_npz_model = NN(
    DNN=nn_npz_DNN,
    num_layers=num_layers_gelu,
    num_neurons=num_neurons_gelu,
    activation_function=activation_function,
    learning_rate=learning_rate_gelu,
    dtype=dtype
)

# Step 3: Load the model weights from the checkpoint
nn_npz_model.load_state_dict(nn_npz_checkpoint["state_dict"])

# Step 4: Optional – move to float64
nn_npz_model = nn_npz_model.to(dtype=dtype, device="cpu")
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

num_jobs = int(os.getenv('NUM_JOBS', '4'))
# === Worker Function ===
def nn_worker(i, x0_np, nd_ntot, truth_t, total_workers):
    global nn_npz_model
    torch.set_num_threads(1)
    device = next(nn_npz_model.parameters()).device

    x0_tensor = torch.tensor(x0_np / nd_ntot, dtype=dtype, device=device)

    start = time.time()
    pred_tensor, _ = forward_nn(nn_npz_model, x0_tensor, truth_t, dtype)
    pred = pred_tensor.detach().cpu().numpy() * nd_ntot
    end = time.time()

    if (i % 1000 == 0) or (i >= total_workers - 100 and i % 10 == 0) or (i == total_workers - 1):
        elapsed = end - start
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"Completed worker {i+1}/{total_workers} "
            f"({elapsed:.2f}s)"
        )
        sys.stdout.flush()

    return i, x0_np, pred, end - start


# Convert entire list to array once (faster indexing, avoid zip unpacking)
initial_guesses_array = np.array(initial_guesses)  # shape (B, 3)
start_time = time.time()
print(f"Start time: {datetime.datetime.fromtimestamp(start_time)}")
# Spawn processes with model initialized globally
print("Torch threads per process:", torch.get_num_threads())
nn_npz_xb0_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=1)(
    delayed(nn_worker)(i, x0_np, nd_ntot, truth_t, len(initial_guesses_array))
    for i, x0_np in enumerate(initial_guesses_array)
)
end_time = time.time()
nn_prediction_loop_time = end_time - start_time
print(f"Running time: {nn_prediction_loop_time:.2f}")
print(f"nn Forward Prediction Complete - {len(nn_npz_xb0_results)} trajectories computed.")
# === Rebuild Dictionary ===
### NN-NPZ Background Results
# === Store NN-NPZ background results ===
xb_dict = {
    i: {
        "truth": truth,
        "perturbed_initial_state": xb_0,
        "nn_npz_background_state": pred,
        "nn_npz_run_time": duration,
    }
    for i, xb_0, pred, duration in nn_npz_xb0_results
}


# === Save the combined dictionary ===
xb_output_path = os.path.join(
    OUTPUT_DIR,
    f"two_samples_per_day_nn_npz_xb_data_"
    f"{NUM_XB0}xb_estimates_"
    f"min_guess_{MIN_GUESS_VAL}_"
    f"{NUM_LAYERS}_{NUM_NEURONS}_compressed.pkl"
)

joblib.dump(xb_dict, xb_output_path)
print(f"Xb data saved to: {xb_output_path}")
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

### nn Jacobian Calculation
# Prepare shared arguments
nn_npz_shared_args = dict(
    model=nn_npz_model,
    dtype=dtype,
    nd_ntot=nd_ntot,
)

# Run in parallel
nn_npz_jacobian_start_time = time.time()
nn_npz_jacobian_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=10)(
    delayed(compute_jacobians)(
        model=nn_npz_shared_args["model"],                        # <-- pass model explicitly
        nd_trajectory=xb_data["nn_npz_background_state"] / nd_ntot,  # rescale to ND
        nd_ntot=nd_ntot,
        return_dimensional=True,                                  # get dimensional Jacobians
        dtype=dtype,
        device=device
    )
    for key, xb_data in xb_dict.items()
)
nn_npz_jacobian_end_time = time.time()

# Convert results back to dictionary (preserve key association)
nn_npz_jacobians_per_trajectory = {
    key: jacobians
    for (key, _), jacobians in zip(xb_dict.items(), nn_npz_jacobian_results)
}

print(f"Jacobian Matrix Run time for all matrices (parallel): "
      f"{nn_npz_jacobian_end_time - nn_npz_jacobian_start_time:.2f}s")
sys.stdout.flush()

# NN-NPZ Assimilation
def run_nn_assimilation_for_trajectory(
    trajectory_key, xb_data, jacobians,
    model_name, model,
    *,  # everything after this must be keyword
    device, dtype, nd_ntot,
    obs_idx, obs_type, obs_time,
    obs_value, obs_error,
    B0, truth_t, nobs, num_cg_iterations):
    
    # Print progress every 1000 samples (rank-independent)
    if trajectory_key % 1000 == 0:
        print(f"Completed assimilation for {trajectory_key} trajectories.")

    xb_0 = xb_data["perturbed_initial_state"]
    xb_model = xb_data[f"{model_name}_background_state"]   # <-- flexible
    truth   = xb_data["truth"]
    
    jo_b_nn = np.sum(((xb_model[obs_idx, obs_type] - obs_value) ** 2) / obs_error)
    jb_b_nn = np.sum(((xb_model[0, :] - xb_0) ** 2) / np.diag(B0))
    J_total_b_nn = 0.5 * jb_b_nn + 0.5 * jo_b_nn

    cg_iter_count = [0]

    def callback(xk):
        cg_iter_count[0] += 1

    # Compute innovation vector (obs - background)
    b = obs_value - xb_model[obs_idx, obs_type]

    innerloop_nn = make_innerloop_nn(
        precomputed_jacobians=jacobians,
        state_matrix_nd_tensor=xb_model,
    )
    np_dtype = np.float64
    # Wrap innerloop in a LinearOperator
    A_orig = LinearOperator(
        shape=(nobs, nobs),
        matvec=innerloop_nn,
        dtype=np_dtype
    )
    
    start = time.time()
    x_orig, exit_code = cg(
        A_orig,
        b,
        rtol=1e-13,
        maxiter=num_cg_iterations,
        callback=callback
    )
    nn_total_time = time.time() - start

    # Step 1: Forcing aligned with model time (already 3D)
    tfrc, frc = obs_forcing(obs_time, obs_type, x_orig, truth_t)  # frc shape = (num_states, 3)
    frc_ad_np = frc   # adjoint forcing is 3D

    num_states = xb_model.shape[0]
    
    # Step 2: Run adjoint with nn (3D Jacobians, 3D forcing)
    ad_nn = propagate_adjoint(
        precomputed_jacobians=[J[:3, :3] for J in jacobians],  # 3x3 Jacobians
        frc_ad_np=frc_ad_np,
        num_states=num_states,
        num_features=3,   # adjoint dimension = 3
        dtype=dtype,
        device=device
    )
    
    # Step 3: Background covariance application (state correction)
    z_nn = B0.dot(ad_nn.T).T    # (num_states, 3)
    z_nn = z_nn[0, :]           # extract initial-time correction
    
    # Step 4: Update analysis initial condition
    xa_0 = torch.tensor((xb_0 + z_nn)/nd_ntot, device=device, dtype=dtype)
    xa_nn_nd, _ = forward_nn(model=model, nd_initial_state=xa_0, trajectory_times=truth_t, dtype=dtype)
    xa_nn = xa_nn_nd.detach().cpu().numpy() * nd_ntot

    jo_a_nn = np.sum(((xa_nn[obs_idx, obs_type] - obs_value)**2) / obs_error)
    jb_a_nn = np.sum(((xa_nn[0, :] - xb_0) ** 2) / np.diag(B0))
    J_total_a_nn = 0.5 * jb_a_nn + 0.5 * jo_a_nn

    misfitb_nn = np.sqrt(np.sum((truth - xb_model) ** 2))
    misfita_nn = np.sqrt(np.sum((truth - xa_nn) ** 2))
    improvement_nn = 100 * ((misfitb_nn - misfita_nn) / misfitb_nn)
    
    pct_drop_J  = safe_pct_drop(J_total_b_nn, J_total_a_nn)
    pct_drop_Jo = safe_pct_drop(jo_b_nn, jo_a_nn)
    pct_drop_Jb = safe_pct_drop(jb_b_nn, jb_a_nn)
    
    result_dict = {
        f"xa_{model_name}": xa_nn,
        f"xb_{model_name}": xb_model,
        f"jo_b_{model_name}": jo_b_nn,
        f"jb_b_{model_name}": jb_b_nn,
        f"J_total_b_{model_name}": J_total_b_nn,
        f"jo_a_{model_name}": jo_a_nn,
        f"jb_a_{model_name}": jb_a_nn,
        f"J_total_a_{model_name}": J_total_a_nn,
        f"{model_name}_misfitb": misfitb_nn,
        f"{model_name}_misfita": misfita_nn,
        f"Improvement_{model_name}": improvement_nn,
        f"pct_drop_J_{model_name}": pct_drop_J,
        f"pct_drop_Jo_{model_name}": pct_drop_Jo,
        f"pct_drop_Jb_{model_name}": pct_drop_Jb,
        f"cg_iterations_{model_name}": cg_iter_count[0],
        f"converged_{model_name}": int(exit_code == 0),
        f"exit_code_{model_name}": exit_code,
        f"{model_name}_time": nn_total_time,
    }

    return trajectory_key, result_dict
   
nn_npz_shared_args = {
    "dtype": dtype,
    "nd_ntot": nd_ntot,
    "obs_time": obs_time,
    "obs_type": obs_type,
    "obs_value": obs_value,
    "obs_error": obs_error,
    "nobs": nobs,
    "B0": B0,
    "model_name": "nn_npz",
    "model": nn_npz_model,
    "device": device,
    "truth_t": truth_t,
    "obs_idx": obs_idx,
    "num_cg_iterations": num_cg_iterations
    
}
nn_npz_assimilation_results_parallel = Parallel(n_jobs=num_jobs, backend="loky")(
    delayed(run_nn_assimilation_for_trajectory)(
        key,
        xb_dict[key],
        nn_npz_jacobians_per_trajectory[key],
        **nn_npz_shared_args
    )
    for key in xb_dict.keys()
)

# Rebuild dict
nn_npz_assimilation_results = {key: result for key, result in nn_npz_assimilation_results_parallel}
print(f"NN-NPZ Assimilation complete - {len(nn_npz_assimilation_results)} trajectories computed.")
assim_output_path = os.path.join(
    OUTPUT_DIR,
    f"two_samples_per_day_nn_npz_frozen_assimilation_"
    f"{NUM_XB0}xb_estimates_"
    f"min_guess_{MIN_GUESS_VAL}_"
    f"{num_cg_iterations}_"
    f"{NUM_LAYERS}_{NUM_NEURONS}_compressed.pkl"
)

joblib.dump(nn_npz_assimilation_results, assim_output_path)
print(f"NN-NPZ assimilation results saved to: {assim_output_path}")
