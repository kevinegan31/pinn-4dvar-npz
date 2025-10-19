'''Pytorch Lightning module for training the NPZ model'''
# Import necessary libraries
import torch
from torch import nn
import lightning as pl
from collections import OrderedDict

# Neural network class
class DNN(nn.Module):
    def __init__(self, layers, activation_function=nn.Tanh):
        super(DNN, self).__init__()
        # Network depth
        self.depth = len(layers) - 1
        # Setup model layers, fully connected + activation function
        layer_list = []
        for i in range(self.depth):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            if i < self.depth - 1:
                # Only apply activation to hidden layers, not after the final output layer
                layer_list.append((f'activation_{i}', activation_function()))
        # Append the final layer separately
        # layer_list.append(('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1])))
        layer_dict = OrderedDict(layer_list)
        self.layers = nn.Sequential(layer_dict)
        # Apply Glorot (Xavier) Initialization
        self.apply(self._initialize_weights)
    def forward(self, x):
        out = self.layers(x)
        return out
    def _initialize_weights(self, m):
        '''Xavier/Glorot initialization for weights and zeros for biases'''
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class PhysicsInformedNN(pl.LightningModule):
    def __init__(self, DNN, num_layers, num_neurons,
                 activation_function, params_dict,
                 lambda_u, lambda_f, learning_rate, dtype):
        '''Physics-informed neural network module'''
        super().__init__()
        self.my_dtype = dtype
        layers = [4] + [num_neurons] * num_layers + [3]
        self.dnn = DNN(layers, activation_function)
        for name, tensor in params_dict.items():
            self.register_buffer(name, tensor)
        self.lambda_u = lambda_u
        self.lambda_f = lambda_f
        self.learning_rate = learning_rate
        self.params_dict = params_dict
    def get_params(self, *param_names):
        return [getattr(self, name) for name in param_names]
    # Training step
    def training_step(self, batch, batch_idx):
        x, u = batch
        # Time step tensor (for automatic differentiation)
        t_step = torch.ones(x.shape[0], 1, dtype=self.my_dtype, device=x.device, requires_grad=True)
        u_pred = self.net_u(x, t_step)
        # Data loss
        loss_u = torch.mean((u - u_pred) ** 2)
        # f_pred = self.net_f(x)
        f_pred = self.net_f(u_pred, t_step)
        loss_f = torch.mean(f_pred ** 2)
        
        loss = self.lambda_u * loss_u + self.lambda_f * loss_f
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('train_loss_u', loss_u, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)  # One-step loss
        self.log('train_loss_f', loss_f, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)  # Physics loss
        return loss
    def validation_step(self, batch, batch_idx):
        x, u = batch
        # Create dummy time (for autograd)
        t_step = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype, requires_grad=True)
        # Forward once
        u_pred = self.net_u(x, t_step)
        # Physics residual
        f_pred = self.net_f(u_pred, t_step)   # assumes you updated net_f to accept u_pred
        # Losses
        loss_u = torch.mean((u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        val_loss = self.lambda_u * loss_u + self.lambda_f * loss_f
        # Logging
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_loss_u", loss_u, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log("val_loss_f", loss_f, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return val_loss
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

        # Exponential decay scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.999  # decay factor per step/epoch
            ),
            "interval": "epoch", # Apply once per epoch
            "name": "lr-Adam"
        }

        return [optimizer], [scheduler]
    # Define the neural network for u
    def net_u(self, x, t=None):
        '''using neural networks to predict u over time t'''
        if t is None:
            t = torch.ones(x.shape[0], 1, requires_grad=True, device=x.device, dtype=self.my_dtype)  # Default time if t is not provided
        input_tensor = torch.cat([x, t], dim=1)
        u = self.dnn(input_tensor)
        return u
    # Non-dimensionalized system of ODEs
    def net_f(self, u_pred, t_step):
        '''NPZ-informed non-dimensionalized function'''
        n_pred, p_pred, z_pred = u_pred[:, 0:1], u_pred[:, 1:2], u_pred[:, 2:3]
        # Retrieve parameters
        alpha_tilde, beta_tilde, b_tilde, c_tilde, e_tilde, f_tilde = self.get_params(
            'alpha_tilde', 'beta_tilde',
            'b_tilde', 'c_tilde',
            'e_tilde', 'f_tilde')
        # Compute time derivatives using autograd (non-dimensionalized and scaled)
        net_dndt = torch.autograd.grad(n_pred, t_step,
                                       grad_outputs=torch.ones_like(n_pred),
                                       retain_graph=True, create_graph=True)[0]
        net_dpdt = torch.autograd.grad(p_pred, t_step,
                                       grad_outputs=torch.ones_like(p_pred),
                                       retain_graph=True, create_graph=True)[0]
        net_dzdt = torch.autograd.grad(z_pred, t_step,
                                       grad_outputs=torch.ones_like(z_pred),
                                       retain_graph=True, create_graph=True)[0]
        # Non-dimensional equations and scaled equations
        pf = (net_dpdt
              - (alpha_tilde * torch.tanh(beta_tilde * n_pred) * p_pred)
              + (b_tilde * p_pred)
              + ((e_tilde + f_tilde) * (p_pred * z_pred)))
        zf = (net_dzdt
              - (e_tilde * (p_pred * z_pred))
              + (c_tilde * z_pred))
        nf = (net_dndt
              + (alpha_tilde * torch.tanh(beta_tilde * n_pred) * p_pred)
              - (b_tilde * p_pred)
              - (c_tilde * z_pred)
              - (f_tilde * (p_pred * z_pred)))
        return torch.cat([nf, pf, zf], dim=1)
    def predict(self, x, t=None):
        '''Predict u and f for inference'''
        self.dnn.eval()

        if t is None:
            t = torch.ones(x.shape[0], 1, dtype=self.my_dtype, device=x.device)

        # Forward once with autograd disabled
        with torch.no_grad():
            u_pred = self.net_u(x, t)

        # For physics residual you need grads, so create a t_step with requires_grad
        t_step = torch.ones(x.shape[0], 1, dtype=self.my_dtype, device=x.device, requires_grad=True)
        u_pred_for_f = self.net_u(x, t_step)
        f_pred = self.net_f(u_pred_for_f, t_step).detach().cpu().numpy()

        return u_pred.cpu().numpy(), f_pred

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


### PINN FUNCTIONS -----------------------------------------
# Forward Model PINN (Uses PyTorch tensors throughout)
def forward_pinn_assimilation(model, nd_initial_state, trajectory_times):
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
