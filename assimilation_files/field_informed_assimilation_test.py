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
sys.path.append('/share/tempest2/egank31/pinn_model_utils/')
from pl_model_utils_frozen import DNN as pi_npz_DNN
from pl_model_utils_frozen import PhysicsInformedNN
# from pl_model_utils_franks import DNN as franks_DNN
# from pl_model_utils_franks import PhysicsInformedNN as franks_PhysicsInformedNN
# from pl_reg_nn import DNN as reg_DNN
# from pl_reg_nn import NN as reg_NN

warnings.filterwarnings('ignore')

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

### Necessary Functions ----------------------  
# ODE NPZ
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

def npz_tl(tl_x, x, t, Vm, ks, m, Rm, ivlev, gamma, q):
    x = np.asarray(x)  # Base state (N, P, Z)
    tl_dx = np.zeros(x.shape)  # Initialize perturbations derivatives (delta N, delta P, delta Z)

    # Derivative of P
    tl_dx[1] = (
        ((Vm * x[0]) / (ks + x[0])) * tl_x[1] +
        ((Vm * ks * x[1]) / ((ks + x[0])**2)) * tl_x[0] - 
        m * tl_x[1] -
        (ivlev * Rm * x[2] * ((1 - np.exp(-ivlev * x[1])) + (ivlev * np.exp(-ivlev * x[1]) * x[1]))) * tl_x[1] -
        (ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * tl_x[2])

    # Derivative of Z
    tl_dx[2] = (
        (ivlev * (1 - gamma) * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * tl_x[2] +
        (ivlev * (1 - gamma) * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * x[1] * np.exp(-ivlev * x[1]))) * tl_x[1] -
        q * tl_x[2]  # Zooplankton mortality
    )

    # Derivative of N
    tl_dx[0] = (
        -(((Vm * x[0]) / (ks + x[0])) * tl_x[1] +
        ((Vm * ks * x[1]) / ((ks + x[0])**2)) * tl_x[0]) + 
        m * tl_x[1] +
        q * tl_x[2] +
        (gamma * ivlev * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * np.exp(-ivlev * x[1]) * x[1])) * tl_x[1] +
        (gamma * ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1])) * tl_x[2])
    )
    return tl_dx

def rk4_tl(f, tl_f, x, tl_x0, times, tfrc, frc, args):
    """
    Perform the gradient integration using the tangent-linear of Runge-Kutta 4. This
    allows for the function to be forced at given times, ft, by the forcing, f.
    """
    nt = len(times)
    tl_x = np.zeros((nt, len(tl_x0)))
    tl_x[0, :] = tl_x0
    dt = np.zeros(nt)
    dt[1:] = np.diff(times)
    for n in range(1, nt):
        k1 = f(x[n - 1, :], times[n], *args) * dt[n]
        k2 = f(x[n - 1, :] + 0.5 * k1, times[n] + 0.5 * dt[n], *args) * dt[n]
        k3 = f(x[n - 1, :] + 0.5 * k2, times[n] + 0.5 * dt[n], *args) * dt[n]
        k4 = f(x[n - 1, :] + k3, times[n] + dt[n], *args) * dt[n]

        tl_k1 = tl_f(tl_x[n - 1, :], x[n - 1, :], times[n], *args) * dt[n]
        tl_k2 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k1,
                     x[n - 1, :] + 0.5 * k1, times[n], * args) * dt[n]
        tl_k3 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k2,
                     x[n - 1, :] + 0.5 * k2, times[n], * args) * dt[n]
        tl_k4 = tl_f(tl_x[n - 1, :] + tl_k3, x[n - 1, :] +
                     k3, times[n] + dt, * args) * dt[n]
        tl_x[n, :] = tl_x[n - 1, :] + \
            (tl_k1 + 2 * tl_k2 + 2 * tl_k3 + tl_k4) / 6

        # Apply the forcing if it is at a forcing time
        if tfrc is not None:
            fl = np.where(np.logical_and(tfrc >= times[n] - 0.5 * dt[n],
                                         tfrc < times[n] + 0.5 * dt[n]))[0]
            tl_x[n, :] += frc[fl, :].sum(axis=0)

    return tl_x
    
# Adjoint Model Code
def npz_ad(ad_x, x, t, Vm, ks, m, Rm, ivlev, gamma, q):
    # Initialize the adjoint derivative array
    ad_dx = np.zeros(x.shape)

    # Adjoint equation for N
    ad_dx[1] -= ((Vm * x[0]) / (ks + x[0])) * ad_x[0]  # Nutrient uptake (affecting P)
    ad_dx[0] -= ((Vm * ks * x[1]) / ((ks + x[0])**2)) * ad_x[0]  # Nutrient limitation (affecting N)
    ad_dx[1] += m * ad_x[0]  # Phytoplankton mortality
    ad_dx[2] += q * ad_x[0]  # Zooplankton mortality

    ad_dx[1] += (gamma * ivlev * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * np.exp(-ivlev * x[1]) * x[1])) * ad_x[0]  # Grazing on P
    ad_dx[2] += (gamma * ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * ad_x[0]  # Grazing on Z
    
    # Adjoint equation for Z
    ad_dx[2] += (ivlev * (1 - gamma) * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * ad_x[2]
    ad_dx[1] += (ivlev * (1 - gamma) * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * x[1] * np.exp(-ivlev * x[1]))) * ad_x[2]
    ad_dx[2] -= q * ad_x[2]  # Zooplankton mortality

    # Adjoint equation for P
    ad_dx[1] += ((Vm * x[0]) / (ks + x[0])) * ad_x[1]  # Nutrient uptake (affecting P)
    ad_dx[0] += ((Vm * ks * x[1]) / ((ks + x[0])**2)) * ad_x[1]  # Nutrient limitation (affecting N)
    ad_dx[1] -= m * ad_x[1]  # Phytoplankton mortality

    ad_dx[1] -= (ivlev * Rm * x[2] * ((1 - np.exp(-ivlev * x[1])) + ivlev * np.exp(-ivlev * x[1]) * x[1])) * ad_x[1]  # P grazing by Z
    ad_dx[2] -= (ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * ad_x[1]  # Z grazing on P



    return ad_dx

def rk4_ad(f, ad_f, x, ad_x0, times, tfrc, frc, args):
    """
    Perform the adjoint integration using the adjoing of Runge-Kutta 4. This
    allows for the function to be forced at given times, ft, by the forcing, f.
    """
    nt = len(times)
    ad_x = np.zeros((nt, len(x0)))
    ad_x[-1, :] = ad_x0
    dt = np.zeros(nt)
    dt[1:] = np.diff(times)
    for n in range(nt - 1, 0, -1):
        k1 = f(x[n - 1, :], times[n], *args) * dt[n]
        k2 = f(x[n - 1, :] + 0.5 * k1, times[n] + 0.5 * dt[n], *args) * dt[n]
        k3 = f(x[n - 1, :] + 0.5 * k2, times[n] + 0.5 * dt[n], *args) * dt[n]
        k4 = f(x[n - 1, :] + k3, times[n] + dt[n], *args) * dt[n]

        # Check if we are at a forcing-time. If so, add the forcing to the
        # solution.
        # if tfrc is not None:
        #     fl = np.where(np.logical_and(tfrc >= times[n] - 0.5 * dt[n],
        #                                  tfrc < times[n] + 0.5 * dt[n]))[0]
        #     ad_x[n - 1, :] += frc[fl, :].sum(axis=0)
        if tfrc is not None:
            ad_x[n - 1, :] += frc[n - 1, :]

        # final-step
        ad_k1 = ad_x[n, :] / 6
        ad_k2 = 2 * ad_x[n, :] / 6
        ad_k3 = 2 * ad_x[n, :] / 6
        ad_k4 = ad_x[n, :] / 6
        ad_x[n - 1, :] += ad_x[n, :]

        # k4-step
        ad_x[n - 1, :] += ad_f(ad_k4 * dt[n], x[n - 1, :] +
                               k3, times[n] + dt[n], *args)
        ad_k3 += ad_f(ad_k4 * dt[n], x[n - 1, :] + k3, times[n] + dt[n], *args)

        # k3-step
        ad_x[n - 1, :] += ad_f(ad_k3 * dt[n], x[n - 1, :] +
                               0.5 * k2, times[n] + 0.5 * dt[n], *args)
        ad_k2 += ad_f(ad_k3 * 0.5 * dt[n], x[n - 1] +
                      0.5 * k2, times[n] + 0.5 * dt[n], *args)

        # k2-step
        ad_x[n - 1, :] += ad_f(ad_k2 * dt[n], x[n - 1, :] +
                               0.5 * k1, times[n] + 0.5 * dt[n], *args)
        ad_k1 += ad_f(ad_k2 * 0.5 * dt[n], x[n - 1, :] +
                      0.5 * k1, times[n] + 0.5 * dt[n], *args)

        # k1-step
        ad_x[n - 1, :] += ad_f(ad_k1 * dt[n], x[n - 1, :], times[n], *args)

        # tl_k1 = tl_f(tl_x[n - 1, :], x[n - 1, :], times[n], *args) * dt[n]
        # tl_k2 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k1,
        #              x[n - 1, :] + 0.5 * k1, times[n], * args) * dt[n]
        # tl_k3 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k2,
        #              x[n - 1, :] + 0.5 * k2, times[n], * args) * dt[n]
        # tl_k4 = tl_f(tl_x[n - 1, :] + tl_k3, x[n - 1, :] +
        #              k3, times[n] + dt, * args) * dt[n]
        # tl_x[n, :] = tl_x[n - 1, :] + (tl_k1 + 2 * tl_k2 + 2 * tl_k3 + tl_k4) / 6

    return ad_x

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

### PINN FUNCTIONS -----------------------------------------
# Forward Model PINN (Uses PyTorch tensors throughout)
def forward_pinn(model, nd_initial_state, trajectory_times):
    # Initialize predictions list
    forward_predictions = []
    initial_state_x0 = nd_initial_state  # Non-dimensionalized initial state
    forward_predictions.append(initial_state_x0)  # Start with initial conditions
    
    # Iterate through time steps
    for i in range(len(trajectory_times) - 1):
        # Use the most recent prediction
        inp = forward_predictions[-1].unsqueeze(0)  # Add batch dimension
        model.dnn.eval()

        # Predict using the model
        ut, _ = model.predict(inp)
        
        # Ensure the output is a PyTorch tensor
        if not isinstance(ut, torch.Tensor):
            ut = torch.tensor(ut, dtype=dtype, device='cpu')
        
        if torch.any(torch.abs(ut) > 1e6):
            print(f"[DEBUG] Explosion at step {i}, state={ut.detach().cpu().numpy()}")
            break

        # === Sanity checks ===
        if torch.any(torch.isnan(ut)) or torch.any(torch.isinf(ut)):
            print(f"[DEBUG] NaN/Inf at step {i}, state={forward_predictions[-1].detach().cpu().numpy()}")
            break

        if torch.any(ut < 0):
            print(f"[DEBUG] Negative values at step {i}, state={ut.detach().cpu().numpy()}")
            break
        # Append predictions to the forward list
        forward_predictions.append(ut.squeeze())
        
    # Stack predictions into a single tensor
    forward_predictions_tensor = torch.stack(forward_predictions)

    return forward_predictions_tensor, trajectory_times

def compute_jacobians(model, nd_trajectory, nd_ntot, return_dimensional=False,
                      dtype=torch.float32, device='cpu'):
    """
    Compute Jacobians of model.dnn along a given ND trajectory.
    Returns ND Jacobians by default, optionally dimensional.
    """
    model.dnn.eval()

    # Append ones column for bias/time feature
    state_matrix_nd = torch.as_tensor(nd_trajectory, dtype=dtype, device=device)
    ones_column = torch.ones((state_matrix_nd.shape[0], 1), dtype=dtype, device=device)
    state_matrix_nd = torch.cat([state_matrix_nd, ones_column], dim=1)

    jacobians = []
    for state in state_matrix_nd:
        F_nd = torch.autograd.functional.jacobian(lambda x: model.dnn(x), state)
        zero_row = torch.zeros((1, F_nd.shape[1]), dtype=F_nd.dtype, device=F_nd.device)
        F_nd = torch.cat([F_nd, zero_row], dim=0)

        if return_dimensional:
            # Ensure nd_ntot is a vector
            if np.isscalar(nd_ntot) or (isinstance(nd_ntot, torch.Tensor) and nd_ntot.ndim == 0):
                nd_ntot_vec = torch.full((F_nd.shape[0]-1,), float(nd_ntot), dtype=dtype, device=device)
            else:
                nd_ntot_vec = torch.as_tensor(nd_ntot, dtype=dtype, device=device)
        
            D = torch.diag(nd_ntot_vec)
            D_inv = torch.linalg.inv(D)
        
            # Only scale the state block (exclude bias row/col)
            F_dim = F_nd.clone()
            F_dim[:-1, :-1] = D @ F_nd[:-1, :-1] @ D_inv
            F_nd = F_dim

        # print(F_nd)


        jacobians.append(F_nd.squeeze(0))

    return jacobians

# TLM Function
def propagate_tlm(precomputed_jacobians, tl_x0, forcing_matrix_tlm, 
                  num_states, num_features, dtype, device):
    predicted_tlms = torch.zeros((num_states, num_features), dtype=dtype, device=device)

    # initial perturbation with bias
    initial_perturbation = torch.cat([
        torch.as_tensor(tl_x0, dtype=dtype, device=device),
        torch.tensor([0.0], dtype=dtype, device=device)
    ])
    predicted_tlms[0] = initial_perturbation

    for i in range(1, num_states):
        jacobian = precomputed_jacobians[i-1]
        updated_perturbation = predicted_tlms[i-1] + forcing_matrix_tlm[i-1]
        predicted_tlms[i] = torch.matmul(jacobian, updated_perturbation)

    return predicted_tlms

# Adjoint Function
def propagate_adjoint(precomputed_jacobians, frc_ad_np, num_states,
                      num_features, dtype, device, lambda_T=None):
    """
    Propagate adjoint backwards (PINN version).

    If lambda_T is provided, uses it as terminal condition.
    Otherwise, runs forcing-driven, zero-terminal adjoint.
    """
    forcing_matrix = torch.tensor(frc_ad_np, dtype=dtype, device=device)
    forcing_matrix_reversed = forcing_matrix.flip(dims=[0])

    # adjoint is only 3D (ignore bias dimension)
    predicted_ad = torch.zeros((num_states, 3), dtype=dtype, device=device)

    if lambda_T is not None:
        # set terminal condition directly
        predicted_ad[-1, :len(lambda_T)] = torch.as_tensor(lambda_T, dtype=dtype, device=device)
        # integrate backward
        for i in reversed(range(num_states-1)):
            jacobian_3x3 = precomputed_jacobians[i][:3, :3]  # ignore bias
            predicted_ad[i] = torch.matmul(jacobian_3x3.T, predicted_ad[i+1]) + forcing_matrix[i]
    else:
        # zero-terminal, forcing-driven (old behavior)
        for i in range(num_states - 1):
            jacobian_3x3 = precomputed_jacobians[num_states - 1 - i][:3, :3]  # ignore bias
            predicted_ad_with_forcing = predicted_ad[i] + forcing_matrix_reversed[i]
            predicted_ad[i + 1] = torch.matmul(jacobian_3x3.T, predicted_ad_with_forcing)
        predicted_ad = predicted_ad.flip(dims=[0])

    return predicted_ad

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

# Set the seed for reproducibility
np.random.seed(42)
### Load in Network ----------------------
# num_threads = torch.get_num_threads()
# torch.set_num_threads(num_threads)

# CPU device
dtype = torch.float64
device = torch.device('cpu') # Smaller models/data, running on CPU

# Load the checkpoint file path
# gelu_checkpoint_path = "../../checkpoints/evolved_states_optuna_trial_0_ntot_500000_nobs_batch_size_64_layers_4_neurons_128_lr_0.001_activation_gelu_lu_1.0_lf_1.0_20_patience_2.9_Nt_10_min.ckpt"
pi_npz_model_ckpt_name = 'optimal_5000_epochs_retrain_opt_01_dataset_500000_nobs_batch_size_2048_layers_5_neurons_384_lr_0.0006_activation_gelu_2.75_Nt_1234_seed_10_min_final.ckpt'
pi_npz_checkpoint_path = f"/share/tempest2/egank31/pinn_tests/assimilation_checkpoints/{pi_npz_model_ckpt_name}"

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

# Define background error model
# error_rk4 = np.array([xb_error_N, xb_error_P, xb_error_Z])
# background_error_rk4 = np.eye(3) * error_rk4**2  # B_inv (inverse of the error covariance matrix)
# error_pinn_vals = np.array([xb_error_N, xb_error_P, xb_error_Z, 0.0])
# background_error_pinn = np.eye(4) * error_pinn_vals**2 # B_inv

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

# Parameters
# --- Observation selection based on realistic frequencies ---
n_days = 7
all_indices = np.arange(len(truth))
time_days = truth_t

# Sampling settings
obs_N_days = [0, 2, 5]       # Simulated cruise days for N (3x per 7 days)
obs_N_per_day = 10           # Higher-res batch on cruise days
obs_P_per_day = 2            # 2x/day for phytoplankton
obs_Z_days = [1, 4]          # Zooplankton sampled 2x/week
obs_Z_per_day = 2

# Initialize
obs_times_N, obs_times_P, obs_times_Z = [], [], []

# Nitrate sampling
for day in obs_N_days:
    day_mask = (time_days >= day) & (time_days < day + 1)
    day_indices = all_indices[day_mask]
    obs_times_N.extend(np.random.choice(day_indices, size=obs_N_per_day, replace=False))

# Phytoplankton daily sampling
for day in range(n_days):
    day_mask = (time_days >= day) & (time_days < day + 1)
    day_indices = all_indices[day_mask]
    obs_times_P.extend(np.random.choice(day_indices, size=obs_P_per_day, replace=False))

# Zooplankton 2x/week sampling
for day in obs_Z_days:
    day_mask = (time_days >= day) & (time_days < day + 1)
    day_indices = all_indices[day_mask]
    obs_times_Z.extend(np.random.choice(day_indices, size=obs_Z_per_day, replace=False))

# Sort
obs_times_N = np.sort(obs_times_N)
obs_times_P = np.sort(obs_times_P)
obs_times_Z = np.sort(obs_times_Z)

# Confirm observation frequencies
n_obs_N = len(obs_times_N)
n_obs_P = len(obs_times_P)
n_obs_Z = len(obs_times_Z)

# Confirm
print(f"N obs/day: {n_obs_N / n_days}")
print(f"P obs/day: {n_obs_P / n_days}")
print(f"Z obs/day: {n_obs_Z / n_days}")


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
def pinn_worker(i, x0_np, nd_ntot, truth_t, total_workers):
    global pi_npz_model
    torch.set_num_threads(1)
    device = next(pi_npz_model.parameters()).device

    x0_tensor = torch.tensor(x0_np / nd_ntot, dtype=dtype, device=device)

    start = time.time()
    pred_tensor, _ = forward_pinn(pi_npz_model, x0_tensor, truth_t)
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
# pi_npz_xb0_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=10)(
#     delayed(pinn_worker)(i, x0_np, nd_ntot)
#     for i, x0_np in enumerate(initial_guesses_array)
# )
# Spawn processes with model initialized globally
print("Torch threads per process:", torch.get_num_threads())
pi_npz_xb0_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=1)(
    delayed(pinn_worker)(i, x0_np, nd_ntot, truth_t, len(initial_guesses_array))
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
joblib.dump(xb_dict, f"/share/tempest2/egank31/assimilation_results/field_informed_xb_data_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_frozen_params_{NUM_LAYERS}_{NUM_NEURONS}_compressed.pkl")
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
rk4_assimilation_results_parallel = Parallel(n_jobs=num_jobs, backend="loky", batch_size=5)(
    delayed(run_rk4_assimilation_for_trajectory)(key, xb_data, **shared_args)
    for key, xb_data in xb_dict.items()
)

# Reconstruct result dictionary
rk4_assimilation_results = {key: result for key, result in rk4_assimilation_results_parallel}
print(f"RK4 Assimilation complete - {len(rk4_assimilation_results)} trajectories computed.")

joblib.dump(rk4_assimilation_results, f"/share/tempest2/egank31/assimilation_results/field_informed_rk4_assimilation_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_{num_cg_iterations}_compressed.pkl")
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

# ### Franks PINN Jacobian Calculation
# # Prepare shared arguments
# franks_pinn_shared_args = dict(
#     model=franks_pinn_model,
#     dtype=dtype,
#     nd_ntot=nd_ntot,
# )

# # Run in parallel
# franks_pinn_jacobian_start_time = time.time()
# franks_pinn_jacobian_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=10)(
#     delayed(compute_jacobians)(
#         model=franks_pinn_shared_args["model"],                        # <-- explicit
#         nd_trajectory=xb_data["franks_pinn_background_state"] / nd_ntot,  # rescale to ND
#         nd_ntot=nd_ntot,
#         return_dimensional=True,   
#         dtype=franks_pinn_shared_args["dtype"],
#         device=device
#     )
#     for key, xb_data in xb_dict.items()
# )

# franks_pinn_jacobian_end_time = time.time()

# # Convert results back to dictionary (preserve key association)
# franks_pinn_jacobians_per_trajectory = {
#     key: jacobians
#     for (key, _), jacobians in zip(xb_dict.items(), franks_pinn_jacobian_results)
# }

# print(f"Jacobian Matrix Run time for all matrices (parallel): "
#       f"{franks_pinn_jacobian_end_time - franks_pinn_jacobian_start_time:.2f}s")
# sys.stdout.flush()
# ### Regular NN Jacobian Calculation
# # Prepare shared arguments
# reg_nn_shared_args = dict(
#     model=nn_model,
#     dtype=dtype,
#     nd_ntot=nd_ntot,
# )

# # Run in parallel
# reg_nn_jacobian_start_time = time.time()
# reg_nn_jacobian_results = Parallel(n_jobs=num_jobs, backend="loky", batch_size=10)(
#     delayed(compute_jacobians_nn)(
#         model=reg_nn_shared_args["model"],                               # <-- explicit
#         nd_trajectory=xb_data["reg_nn_background_state"] / nd_ntot,      # rescale to ND
#         nd_ntot=nd_ntot,
#         return_dimensional=True,   
#         dtype=reg_nn_shared_args["dtype"],
#         device=device
#     )
#     for key, xb_data in xb_dict.items()
# )
# reg_nn_jacobian_end_time = time.time()

# # Convert results back to dictionary (preserve key association)
# reg_nn_jacobians_per_trajectory = {
#     key: jacobians
#     for (key, _), jacobians in zip(xb_dict.items(), reg_nn_jacobian_results)
# }

# print(f"Jacobian Matrix Run time for all matrices (parallel): "
#       f"{reg_nn_jacobian_end_time - reg_nn_jacobian_start_time:.2f}s")
# sys.stdout.flush()

# PI-NPZ Assimilation
def run_pinn_assimilation_for_trajectory(
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
    
    jo_b_pinn = np.sum(((xb_model[obs_idx, obs_type] - obs_value) ** 2) / obs_error)
    jb_b_pinn = np.sum(((xb_model[0, :] - xb_0) ** 2) / np.diag(B0))
    J_total_b_pinn = 0.5 * jb_b_pinn + 0.5 * jo_b_pinn

    cg_iter_count = [0]

    def callback(xk):
        cg_iter_count[0] += 1

    # Compute innovation vector (obs - background)
    b = obs_value - xb_model[obs_idx, obs_type]

    innerloop_pinn = make_innerloop_pinn(
        precomputed_jacobians=jacobians,
        state_matrix_nd_tensor=xb_model,
    )
    np_dtype = np.float64
    # Wrap innerloop in a LinearOperator
    A_orig = LinearOperator(
        shape=(nobs, nobs),
        matvec=innerloop_pinn,
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
    pinn_total_time = time.time() - start

    # Step 1: Forcing aligned with model time (already 3D)
    tfrc, frc = obs_forcing(obs_time, obs_type, x_orig, truth_t)  # frc shape = (num_states, 3)
    frc_ad_np = frc   # adjoint forcing is 3D

    num_states = xb_model.shape[0]
    
    # Step 2: Run adjoint with PINN (3D Jacobians, 3D forcing)
    ad_pinn = propagate_adjoint(
        precomputed_jacobians=[J[:3, :3] for J in jacobians],  # 3x3 Jacobians
        frc_ad_np=frc_ad_np,
        num_states=num_states,
        num_features=3,   # adjoint dimension = 3
        dtype=dtype,
        device=device
    )
    
    # Step 3: Background covariance application (state correction)
    z_pinn = B0.dot(ad_pinn.T).T    # (num_states, 3)
    z_pinn = z_pinn[0, :]           # extract initial-time correction
    
    # Step 4: Update analysis initial condition
    xa_0 = torch.tensor((xb_0 + z_pinn)/nd_ntot, device=device, dtype=dtype)
    xa_pinn_nd, _ = forward_pinn(model=model, nd_initial_state=xa_0, trajectory_times=truth_t)
    xa_pinn = xa_pinn_nd.detach().numpy() * nd_ntot

    jo_a_pinn = np.sum(((xa_pinn[obs_idx, obs_type] - obs_value)**2) / obs_error)
    jb_a_pinn = np.sum(((xa_pinn[0, :] - xb_0) ** 2) / np.diag(B0))
    J_total_a_pinn = 0.5 * jb_a_pinn + 0.5 * jo_a_pinn

    misfitb_pinn = np.sqrt(np.sum((truth - xb_model) ** 2))
    misfita_pinn = np.sqrt(np.sum((truth - xa_pinn) ** 2))
    improvement_pinn = 100 * ((misfitb_pinn - misfita_pinn) / misfitb_pinn)
    
    pct_drop_J  = safe_pct_drop(J_total_b_pinn, J_total_a_pinn)
    pct_drop_Jo = safe_pct_drop(jo_b_pinn, jo_a_pinn)
    pct_drop_Jb = safe_pct_drop(jb_b_pinn, jb_a_pinn)
    
    result_dict = {
        f"xa_{model_name}": xa_pinn,
        f"xb_{model_name}": xb_model,
        f"jo_b_{model_name}": jo_b_pinn,
        f"jb_b_{model_name}": jb_b_pinn,
        f"J_total_b_{model_name}": J_total_b_pinn,
        f"jo_a_{model_name}": jo_a_pinn,
        f"jb_a_{model_name}": jb_a_pinn,
        f"J_total_a_{model_name}": J_total_a_pinn,
        f"{model_name}_misfitb": misfitb_pinn,
        f"{model_name}_misfita": misfita_pinn,
        f"Improvement_{model_name}": improvement_pinn,
        f"pct_drop_J_{model_name}": pct_drop_J,
        f"pct_drop_Jo_{model_name}": pct_drop_Jo,
        f"pct_drop_Jb_{model_name}": pct_drop_Jb,
        f"cg_iterations_{model_name}": cg_iter_count[0],
        f"converged_{model_name}": int(exit_code == 0),
        f"exit_code_{model_name}": exit_code,
        f"{model_name}_time": pinn_total_time,
    }

    return trajectory_key, result_dict
   
pi_npz_shared_args = {
    "dtype": dtype,
    "nd_ntot": nd_ntot,
    "obs_time": obs_time,
    "obs_type": obs_type,
    "obs_value": obs_value,
    "obs_error": obs_error,
    "nobs": nobs,
    "B0": B0,
    "model_name": "pi_npz",
    "model": pi_npz_model,
    "device": device,
    "truth_t": truth_t,
    "obs_idx": obs_idx,
    "num_cg_iterations": num_cg_iterations
    
}
pi_npz_assimilation_results_parallel = Parallel(n_jobs=num_jobs, backend="loky")(
    delayed(run_pinn_assimilation_for_trajectory)(
        key,
        xb_dict[key],
        pi_npz_jacobians_per_trajectory[key],
        **pi_npz_shared_args
    )
    for key in xb_dict.keys()
)

# franks_pinn_shared_args = {
#     "dtype": dtype,
#     "nd_ntot": nd_ntot,
#     "obs_time": obs_time,
#     "obs_type": obs_type,
#     "obs_value": obs_value,
#     "obs_error": obs_error,
#     "nobs": nobs,
#     "B0": B0,
#     "model_name": "franks_pinn",
#     "model": franks_pinn_model,
#     "device": device,
#     "truth_t": truth_t,
#     "obs_idx": obs_idx,
#     "num_cg_iterations": num_cg_iterations
    
# }
# franks_pinn_assimilation_results_parallel = Parallel(n_jobs=num_jobs, backend="loky")(
#     delayed(run_pinn_assimilation_for_trajectory)(
#         key,
#         xb_dict[key],
#         franks_pinn_jacobians_per_trajectory[key],
#         **franks_pinn_shared_args
#     )
#     for key in xb_dict.keys()
# )

# nn_assimilation_results = {}

# def run_reg_nn_assimilation_for_trajectory(
#     trajectory_key, xb_data, jacobians,
#     model_name, model,
#     *,  # everything after this must be keyword
#     device, dtype, nd_ntot,
#     obs_idx, obs_type, obs_time,
#     obs_value, obs_error,
#     B0, truth_t, nobs, num_cg_iterations):

#     xb_0 = xb_data["perturbed_initial_state"]
#     xb_model = xb_data[f"{model_name}_background_state"]   # <-- flexible
#     truth   = xb_data["truth"]

#     cg_iter_count = [0]

#     def callback(xk):
#         cg_iter_count[0] += 1

#     # Compute innovation vector (obs - background)
#     b = obs_value - xb_model[obs_idx, obs_type]

#     innerloop_nn = make_innerloop_nn(
#         precomputed_jacobians=jacobians,
#         state_matrix_nd_tensor=xb_model,
#     )
#     np_dtype = np.float64
#     # Wrap innerloop in a LinearOperator
#     A_orig = LinearOperator(
#         shape=(nobs, nobs),
#         matvec=innerloop_nn,
#         dtype=np_dtype
#     )
    
#     start = time.time()
#     x_orig, exit_code = cg(
#         A_orig,
#         b,
#         rtol=1e-13,
#         maxiter=num_cg_iterations,
#         callback=callback
#     )
#     nn_total_time = time.time() - start

#     # Step 1: Forcing aligned with model time (already 3D)
#     tfrc, frc = obs_forcing(obs_time, obs_type, x_orig, truth_t)  # frc shape = (num_states, 3)
#     frc_ad_np = frc   # adjoint forcing is 3D

#     num_states = xb_model.shape[0]
    
#     # Step 2: Run adjoint with PINN (3D Jacobians, 3D forcing)
#     ad_nn = propagate_adjoint_nn(
#         precomputed_jacobians=[J[:3, :3] for J in jacobians],  # 3x3 Jacobians
#         frc_ad_np=frc_ad_np,
#         num_states=num_states,
#         num_features=3,   # adjoint dimension = 3
#         dtype=dtype,
#         device=device
#     )
    
#     # Step 3: Background covariance application (state correction)
#     z_nn = B0.dot(ad_nn.T).T    # (num_states, 3)
#     z_nn = z_nn[0, :]           # extract initial-time correction
    
#     # Step 4: Update analysis initial condition
#     xa_0 = torch.tensor((xb_0 + z_nn)/nd_ntot, device=device, dtype=dtype)
#     xa_nn_nd, _ = forward_nn(model=model, nd_initial_state=xa_0, trajectory_times=truth_t)
#     xa_nn = xa_nn_nd.detach().numpy() * nd_ntot

#     jo_a_nn = np.sum(((xa_nn[obs_idx, obs_type] - obs_value)**2) / obs_error)
#     jb_a_nn = np.sum(((xa_nn[0, :] - xb_0) ** 2) / np.diag(B0))
#     J_total_a_nn = 0.5 * jb_a_nn + 0.5 * jo_a_nn

#     misfitb_nn = np.sqrt(np.sum((truth - xb_model) ** 2))
#     misfita_nn = np.sqrt(np.sum((truth - xa_nn) ** 2))
#     improvement_nn = 100 * ((misfitb_nn - misfita_nn) / misfitb_nn)

#     result_dict = {
#         f"xa_{model_name}": xa_nn,
#         f"jo_a_{model_name}": jo_a_nn,
#         f"jb_a_{model_name}": jb_a_nn,
#         f"J_total_a_{model_name}": J_total_a_nn,
#         f"{model_name}_misfitb": misfitb_nn,
#         f"{model_name}_misfita": misfita_nn,
#         f"Improvement_{model_name}": improvement_nn,
#         f"cg_iterations_{model_name}": cg_iter_count[0],
#         f"converged_{model_name}": int(exit_code == 0),
#         f"exit_code_{model_name}": exit_code,
#         f"{model_name}_time": nn_total_time,
#     }

#     return trajectory_key, result_dict

# reg_nn_shared_args = {
#     "dtype": dtype,
#     "nd_ntot": nd_ntot,
#     "obs_time": obs_time,
#     "obs_type": obs_type,
#     "obs_value": obs_value,
#     "obs_error": obs_error,
#     "nobs": nobs,
#     "B0": B0,
#     "model_name": "reg_nn",
#     "model": nn_model,
#     "device": device,
#     "truth_t": truth_t,
#     "obs_idx": obs_idx,
#     "num_cg_iterations": num_cg_iterations
    
# }
# reg_nn_assimilation_results_parallel = Parallel(n_jobs=num_jobs, backend="loky")(
#     delayed(run_reg_nn_assimilation_for_trajectory)(
#         key,
#         xb_dict[key],
#         reg_nn_jacobians_per_trajectory[key],
#         **reg_nn_shared_args
#     )
#     for key in xb_dict.keys()
# )

# Rebuild dict
pi_npz_assimilation_results = {key: result for key, result in pi_npz_assimilation_results_parallel}
print(f"PI-NPZ Assimilation complete - {len(pi_npz_assimilation_results)} trajectories computed.")
joblib.dump(pi_npz_assimilation_results, f"/share/tempest2/egank31/assimilation_results/field_informed_pi_npz_frozen_assimilation_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_{num_cg_iterations}_{NUM_LAYERS}_{NUM_NEURONS}_compressed.pkl")

# franks_pinn_assimilation_results = {key: result for key, result in franks_pinn_assimilation_results_parallel}
# print(f"Franks PINN Assimilation complete - {len(franks_pinn_assimilation_results)} trajectories computed.")
# joblib.dump(franks_pinn_assimilation_results, f"/share/tempest2/egank31/assimilation_results/field_informed_franks_pinn_assimilation_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_{num_cg_iterations}_compressed.pkl")

# nn_assimilation_results = {key: result for key, result in reg_nn_assimilation_results_parallel}
# print(f"Regular NN Assimilation complete - {len(nn_assimilation_results)} trajectories computed.")
# joblib.dump(nn_assimilation_results, f"/share/tempest2/egank31/assimilation_results/field_informed_reg_nn_assimilation_{NUM_XB0}xb_estimates_min_guess_{MIN_GUESS_VAL}_{num_cg_iterations}_compressed.pkl")
