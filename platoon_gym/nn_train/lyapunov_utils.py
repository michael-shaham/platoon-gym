import numpy as np
import os
import torch
from typing import List, Optional, Tuple

from platoon_gym.dyn.linear_vel import LinearVel
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.nn_models.feedforward import FullyConnected


def initialize_dynamics(
    N: int,
    dt: float,
    tau_min: float,
    tau_max: float,
    rng: Optional[np.random.Generator] = None,
) -> List[LinearVel]:
    """
    Create a list of LinearVel objects for N vehicles.
    """
    if rng is None:
        rng = np.random.default_rng()
    taus = rng.uniform(tau_min, tau_max, N)
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    dyns = [LinearVel(dt, x_lims, u_lims, tau) for tau in taus]
    return dyns


def initialize_vehicles(
    N: int,
    dyns: List[LinearVel],
    d_des: List[float],
    v_init: float,
    errs: np.ndarray,
) -> List[Vehicle]:
    """
    Create a list of Vehicle objects for N vehicles.
    """
    vehs = []
    # first vehicle
    p0 = -d_des[0] - errs[0, 0]
    v0 = v_init - errs[0, 1]
    vehs.append(Vehicle(dyns[0], p0, v0))
    print(f"errs=\n{errs}")
    for i in range(1, N):
        p0 = vehs[-1].state[0] - d_des[i] - errs[i, 0]
        v0 = vehs[-1].state[1] - errs[i, 1]
        vehs.append(Vehicle(dyns[i], p0, v0))
    return vehs


def create_control_model(
    ctrl_hidden_dims: List[int],
    act_fn: torch.nn.Module,
    guide_control: bool,
    control_limit: float,
    device: str,
    save_dir: str,
) -> Tuple[FullyConnected, str]:
    """
    Creates the neural network feedback control model. Returns the model and
    the file in which to save the model.
    """
    in_dim = 2
    dims = [in_dim] + ctrl_hidden_dims + [1]
    dims_str = "_".join([str(d) for d in dims])
    if isinstance(act_fn, torch.nn.LeakyReLU):
        act_fn_str = "leaky_relu"
    else:
        act_fn_str = "relu"
    file_name = f"vel_dyn_ctrl_{dims_str}_{act_fn_str}"
    if guide_control:
        file_name += "_guided"
    ctrl_file = os.path.join(save_dir, "models", file_name + ".pt")
    ctrl_min = torch.tensor([-control_limit]).to(device)
    ctrl_max = torch.tensor([control_limit]).to(device)
    ctrl = FullyConnected(
        in_dim,
        1,
        ctrl_hidden_dims,
        ctrl_min.clone(),
        ctrl_max.clone(),
        bias=False,
        act_fn=act_fn,
    )
    ctrl.to(device)
    # if os.path.exists(ctrl_file):
    #     ctrl.load_state_dict(torch.load(ctrl_file))
    return ctrl, ctrl_file


def create_lyapunov_model(
    num_vehicles: int,
    lyap_hidden_dims: List[int],
    act_fn: torch.nn.Module,
    guide_control: bool,
    device: str,
    save_dir: str,
) -> Tuple[FullyConnected, str]:
    """
    Creates the neural network Lyapunov model. Returns the model and the file
    in which to save the model.
    """
    in_dim = 2 * num_vehicles
    dims = [in_dim] + lyap_hidden_dims + [1]
    dims_str = "_".join([str(d) for d in dims])
    lyap = FullyConnected(
        in_dim, 1, hidden_dims=lyap_hidden_dims, bias=False, act_fn=act_fn
    )
    if isinstance(act_fn, torch.nn.LeakyReLU):
        act_fn_str = "leaky_relu"
    else:
        act_fn_str = "relu"
    file_name = f"vel_dyn_lyap_{dims_str}_{act_fn_str}"
    if guide_control:
        file_name += "_guided"
    lyap_file = os.path.join(save_dir, "models", file_name + ".pt")
    lyap.to(device)
    # if os.path.exists(lyap_file):
    #     lyap.load_state_dict(torch.load(lyap_file))
    return lyap, lyap_file


def generate_new_lyapunov_model(
    num_vehicles_previous: int,
    num_vehicles_new: int,
    lyap_hidden_dims: List[int],
    act_fn: torch.nn.Module,
    guide_control: bool,
    device: str,
    save_dir: str,
    prev_lyap_model: FullyConnected,
) -> Tuple[FullyConnected, str]:
    """
    Generates a new Lyapunov model for a different number of vehicles. The new
    model is initialized with (some of) the weights of the previous model.
    """
    lyap_new, lyap_file = create_lyapunov_model(
        num_vehicles_new, lyap_hidden_dims, act_fn, guide_control, device, save_dir
    )
    lyap_new.layers[0].weight.data[:, : 2 * num_vehicles_previous] = (
        prev_lyap_model.layers[0].weight.data[:, : 2 * num_vehicles_previous]
    )
    for layer, layer_new in zip(prev_lyap_model.layers[1:], lyap_new.layers[1:]):
        if isinstance(layer, torch.nn.Linear):
            layer_new.weight.data = layer.weight.data.clone()
    return lyap_new, lyap_file
