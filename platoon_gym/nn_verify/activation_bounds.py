import numpy as np
from typing import List, Tuple

from platoon_gym.nn_models.feedforward import FullyConnected
from platoon_gym.nn_verify.utils import model_to_weights_biases_acts


def ibp_preactivation_bounds(
    model: FullyConnected,
    input_range: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Computes the preactivation bounds for a fully connected ReLU network using
    interval bound propagation. Assumes ReLU activation functions.

    Args:
        Ws: list of weight matrices
        bs: list of bias vectors

    Returns:
        ls: list of lower bounds for each layer
        us: list of upper bounds for each layer
    """
    Ws, bs, relus, lrelus = model_to_weights_biases_acts(model)
    ls, us = [], []
    z_low = input_range[:, 0]
    z_high = input_range[:, 1]
    if len(bs) == 0:
        bs = [np.zeros(W.shape[0]) for W in Ws]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        mu = (z_high + z_low) / 2
        r = (z_high - z_low) / 2
        mu = W @ mu + b
        r = np.abs(W) @ r
        z_low = mu - r
        z_high = mu + r
        if relus[i]:
            ls.append(z_low)
            us.append(z_high)
            z_low = np.maximum(z_low, 0)
            z_high = np.maximum(z_high, 0)
        elif lrelus[i]:
            ls.append(z_low)
            us.append(z_high)
            z_low = np.maximum(z_low, lrelus[i] * z_low)
            z_high = np.maximum(z_high, lrelus[i] * z_high)
        else:
            ls.append(z_low)
            us.append(z_high)
    return ls, us
