import numpy as np
from typing import Any, Dict, List
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """MLP with dynamic number of hidden layers."""
    
    def __init__(
        self,
        data_config: Dict[str, Any],
        hidden_layers: List[int] = [1024, 128],  # Default hidden layers sizes
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dim = np.prod(data_config["input_dims"])
        num_classes = len(data_config["mapping"])
        
        # Creating all layers dynamically
        self.layers = nn.ModuleList()
        last_dim = input_dim

        # Add hidden layers based on the sizes provided in hidden_layers
        for layer_size in hidden_layers:
            self.layers.append(nn.Linear(last_dim, layer_size))
            self.layers.append(nn.Dropout(0.5))  # Adding dropout after each hidden layer
            last_dim = layer_size
        
        # Output layer
        self.layers.append(nn.Linear(last_dim, num_classes))

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x)  # Apply dropout
        x = self.layers[-1](x)  # Output layer, no activation
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--hidden_layers", type=int, nargs='+', default=[1024, 128])
        return parser

### Below this line is for testing
parser = argparse.ArgumentParser()
MLP.add_to_argparse(parser)
args = parser.parse_args('--hidden_layers 512 256 128'.split())

data_config = {
    "input_dims": (1, 28, 28),
    "mapping": [str(i) for i in range(10)]
}

model = MLP(data_config, args.hidden_layers, args)
