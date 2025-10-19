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
