import numpy as np
from torch.nn import Linear, ReLU, LeakyReLU
from typing import List, Tuple, Union

from platoon_gym.nn_models.feedforward import FullyConnected


def model_to_weights_biases_acts(
    m: FullyConnected,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[Union[float, None]]]:
    """
    Extracts the weights and biases from a PyTorch model.

    Args:
        m: FullyConnected model

    Returns:
        weights: list of weight matrices
        biases: list of bias vectors
        relus: list of booleans indicating if the layer is followed by a ReLU
        lrelus: list of floats indicating the slope of the negative portion of
            the Leaky ReLU if the layer is followed by a Leaky ReLU, otherwise
            None
    """
    weights = []
    biases = []
    relus = []
    lrelus = []
    for i, l in enumerate(m.layers):
        if isinstance(l, Linear):
            weights.append(l.weight.detach().cpu().numpy())
            if l.bias is not None:
                biases.append(l.bias.detach().cpu().numpy())
            else:
                biases.append(np.zeros(l.out_features))

            if i + 1 < len(m.layers) and isinstance(m.layers[i + 1], ReLU):
                relus.append(True)
            else:
                relus.append(False)

            if i + 1 < len(m.layers) and isinstance(m.layers[i + 1], LeakyReLU):
                lrelus.append(m.layers[i + 1].negative_slope)
            else:
                lrelus.append(None)

        elif isinstance(l, ReLU) or isinstance(l, LeakyReLU):
            continue

        else:
            raise ValueError(f"Unsupported layer type {type(l)}")

    return weights, biases, relus, lrelus
