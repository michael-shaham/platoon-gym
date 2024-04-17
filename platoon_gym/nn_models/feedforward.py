import torch
from torch import nn

from typing import List, Optional


class FullyConnected(nn.Module):
    """A fully-connected neural network."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        out_min: Optional[torch.Tensor] = None,
        out_max: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        bias: bool = True,
        act_fn: torch.nn.Module = nn.LeakyReLU(),
    ):
        """
        Initialize the structure.

        Args:
            in_dim: the dimension of the input
            out_dim: the dimension of the output
            hidden_dims: a list of hidden dimensions
            out_min: optional torch tensor, shape (out_dim,), the min out value
            out_max: optional torch tensor, shape (out_dim,), the max out value
        """
        super().__init__()
        self.out_dim = out_dim
        self.flatten = nn.Flatten()
        self.act_fn = act_fn
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dims[0], bias=bias))
        self.layers.append(act_fn)
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=bias))
            self.layers.append(act_fn)
        self.layers.append(nn.Linear(hidden_dims[-1], out_dim, bias=bias))
        self.unclamped_length = len(self.layers)

        # if given output min/max, clip the output:
        self.set_clamp_layers(out_min, out_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If out_min and out_max is given, we add two layers to the
        network to clip the output.

        Args:
            x: the input tensor

        Returns:
            the output tensor
        """
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, self.act_fn.__class__):
                x = self.dropout(x)
        return x

    def set_clamp_layers(
        self, out_min: Optional[torch.Tensor], out_max: Optional[torch.Tensor]
    ) -> None:
        """
        Add layers to clip the output if out_min and out_max are given. The
        function clamp(out, out_min, out_max) can be implemented as follows:
            final = out_max - ReLU(out_max - ReLU(out - out_min) - out_min)

        Args:
            out_min: optional torch tensor, shape (out_dim,), the min out value
            out_max: optional torch tensor, shape (out_dim,), the max out value
        """
        self.layers = self.layers[: self.unclamped_length]
        if out_min is None and out_max is None:
            return
        if out_min is None:
            out_min = -torch.ones(self.out_dim) * float("inf")
        if out_max is None:
            out_max = torch.ones(self.out_dim) * float("inf")

        assert out_min.shape == (self.out_dim,)
        assert out_max.shape == (self.out_dim,)

        # u1 = ReLU(uout - umin)
        layer1 = nn.Linear(self.out_dim, self.out_dim)
        layer1.weight.data = torch.eye(self.out_dim)
        layer1.bias.data = -out_min
        layer1.weight.requires_grad = False
        layer1.bias.requires_grad = False
        self.layers.append(layer1)
        self.layers.append(self.relu)

        # u2 = ReLU(umax - u1 - umin)
        layer2 = nn.Linear(self.out_dim, self.out_dim)
        layer2.weight.data = -torch.eye(self.out_dim)
        layer2.bias.data = out_max - out_min
        layer2.weight.requires_grad = False
        layer2.bias.requires_grad = False
        self.layers.append(layer2)
        self.layers.append(self.relu)

        # ufinal = umax - u2
        layer3 = nn.Linear(self.out_dim, self.out_dim)
        layer3.weight.data = -torch.eye(self.out_dim)
        layer3.bias.data = out_max
        layer3.weight.requires_grad = False
        layer3.bias.requires_grad = False
        self.layers.append(layer3)
