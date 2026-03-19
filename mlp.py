import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


def _get_activation_fn(activation):
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Activation function {activation} not supported")


class MlpS(nn.Module):
    """
    MLP as used in MLP-Mixer.
    Structure: Linear -> LN -> Act -> Drop -> Linear (projection)
    """
    def __init__(
        self,
        input_dim=None,
        hidden_dim=None,
        output_dim=None,
        dropout_p=0.1,
        activation_fn="gelu",
        use_layernorm=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim*2

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.act = _get_activation_fn(activation_fn)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MlpL(nn.Module):
    """
    MLP architectures with flexible dimension specifications.
    Can handle expansion (e.g., 512 -> 2048 -> 512 -> 256) or compression patterns.
    Dims are automatically derived from the fc_dims list specified during the init.

    in_features = fc_dims[0]
    out_features = fc_dims[-1]
    hidden_features = fc_dims[1] if len(fc_dims) > 2 else fc_dims[0]
    """
    def __init__(
        self,
        act_layer="gelu",
        drop=0.1,
        activation_fn="gelu",
        fc_dims=None,  # List of dimensions for multi-layer MLP
    ):
        super().__init__()

        # If fc_dims is provided, derive in_features and out_features from it
        if fc_dims is not None:
            assert len(fc_dims) >= 2, "fc_dims must have at least 2 dimensions"
            in_features = fc_dims[0]
            out_features = fc_dims[-1]
            hidden_features = fc_dims[1] if len(fc_dims) > 2 else fc_dims[0]

            layers = []
            for i in range(len(fc_dims) - 1):
                layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))

                # Add activation and dropout to all layers except the last
                if i < len(fc_dims) - 2:
                    layers.append(_get_activation_fn(activation_fn))
                    layers.append(nn.Dropout(drop))

            self.net = nn.Sequential(*layers)
        else:
            raise ValueError("fc_dims cannot be None. Require manadatory args")
        # Store dimensions for reference
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

    def forward(self, emb, valid_mask=None):
        """
        Arg:
            x: [..., :, hidden_dim]
            valid_mask: [B, T, 1]
        returns:
            x: [..., :, out_dim]
        """
        x = self.net(emb)

        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(-1) # [B, T, 1, 1]
            x = x.masked_fill(~valid_mask.bool(), 0.0)

        return x
