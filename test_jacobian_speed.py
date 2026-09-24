import os
import sys
import time
import numpy as np
import torch

# ---------------------------------------------------------
# Repo paths
# ---------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(REPO_ROOT, "models")
sys.path.append(MODEL_DIR)

from nn_npz import (
    NN,
    DNN,
    forward_nn_assimilation,
    compute_jacobians_old,
    compute_jacobians,
)

# ---------------------------------------------------------
# Configuration -- same as production NN assimilation
# ---------------------------------------------------------
dtype = torch.float64
device = "cpu"

num_layers = 5
num_neurons = 512
learning_rate = 0.0001
activation_function = torch.nn.GELU

nd_ntot = 2.75

checkpoint_path = os.path.join(
    REPO_ROOT,
    "model_checkpoints",
    "nn_npz_final.ckpt",
)

# ---------------------------------------------------------
# Load trained NN
# ---------------------------------------------------------
print("Loading model...")

model = NN.load_from_checkpoint(
    checkpoint_path,
    DNN=DNN,
    num_layers=num_layers,
    num_neurons=num_neurons,
    activation_function=activation_function,
    learning_rate=learning_rate,
    dtype=dtype,
    map_location="cpu",
)

model = model.to(dtype=dtype, device=device)
model.eval()

# ---------------------------------------------------------
# Generate one representative 7-day NN trajectory
#
# Use the same 10-minute timestep as the assimilation.
# Initial dimensional state [1.6, 0.3, 0.1].
# Convert to nondimensional state using ntot = 2.75.
# ---------------------------------------------------------
dt_minutes = 10
num_days = 7

num_steps = int(num_days * 24 * 60 / dt_minutes)

trajectory_times = torch.arange(
    num_steps + 1,
    dtype=dtype,
    device=device,
)

x0_dim = torch.tensor(
    [1.6, 0.3, 0.1],
    dtype=dtype,
    device=device,
)

x0_nd = x0_dim / nd_ntot

print("Generating NN trajectory...")

trajectory_nd, _ = forward_nn_assimilation(
    model,
    x0_nd,
    trajectory_times,
    dtype,
)

print("Trajectory shape:", trajectory_nd.shape)

if trajectory_nd.shape[0] != num_steps + 1:
    raise RuntimeError(
        f"Trajectory terminated early: "
        f"{trajectory_nd.shape[0]} states instead of {num_steps + 1}"
    )

# ---------------------------------------------------------
# OLD Jacobian implementation
# ---------------------------------------------------------
print("\nRunning OLD Jacobian implementation...")

t0 = time.perf_counter()

J_old = compute_jacobians_old(
    model,
    trajectory_nd,
    nd_ntot,
    return_dimensional=True,
    dtype=dtype,
    device=device,
)

old_time = time.perf_counter() - t0

J_old = torch.stack(J_old)

print(f"Old runtime: {old_time:.6f} s")
print("Old shape:", J_old.shape)

# ---------------------------------------------------------
# NEW vectorized Jacobian implementation
# ---------------------------------------------------------
print("\nRunning NEW vectorized Jacobian implementation...")

t0 = time.perf_counter()

J_new = compute_jacobians(
    model,
    trajectory_nd,
    nd_ntot,
    return_dimensional=True,
    dtype=dtype,
    device=device,
)

new_time = time.perf_counter() - t0

J_new = torch.stack(J_new)

print(f"New runtime: {new_time:.6f} s")
print("New shape:", J_new.shape)

# ---------------------------------------------------------
# Numerical equivalence
# ---------------------------------------------------------
difference = torch.abs(J_old - J_new)

max_abs_diff = difference.max().item()
mean_abs_diff = difference.mean().item()

allclose = torch.allclose(
    J_old,
    J_new,
    rtol=1e-12,
    atol=1e-12,
)

exact_equal = torch.equal(J_old, J_new)

print("\n======================================")
print("NUMERICAL COMPARISON")
print("======================================")
print("Old shape:       ", J_old.shape)
print("New shape:       ", J_new.shape)
print("Max abs diff:    ", max_abs_diff)
print("Mean abs diff:   ", mean_abs_diff)
print("Exactly equal:   ", exact_equal)
print("Allclose 1e-12:  ", allclose)

# ---------------------------------------------------------
# Speed comparison
# ---------------------------------------------------------
print("\n======================================")
print("SPEED COMPARISON")
print("======================================")
print(f"Old runtime:     {old_time:.6f} s")
print(f"New runtime:     {new_time:.6f} s")

if new_time > 0:
    print(f"Speedup:         {old_time / new_time:.2f}x")

# ---------------------------------------------------------
# Explicitly fail if the implementations disagree
# ---------------------------------------------------------
if not allclose:
    raise RuntimeError(
        "Vectorized Jacobians do NOT agree with the original "
        "implementation at rtol=atol=1e-12."
    )

print("\nPASS: Vectorized Jacobians agree with original implementation.")