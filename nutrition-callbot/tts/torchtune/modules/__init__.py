"""Minimal subset of torchtune.modules used by neucodec.

This local shim avoids importing the full torchtune package, which in some
runtime combinations pulls in torchao with incompatible torch APIs.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """Apply RoPE on the last dimension of a tensor.

    Expected input shape in this project: (batch, heads, seq_len, head_dim).
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("RotaryPositionalEmbeddings dim must be a positive even integer")

        self.dim = dim
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError("Expected tensor with at least 3 dimensions for RoPE")

        if x.shape[-1] < self.dim:
            raise ValueError(f"Last dim {x.shape[-1]} is smaller than configured rotary dim {self.dim}")

        seq_len = x.shape[-2]
        device = x.device
        dtype = x.dtype

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))

        cos = torch.cos(freqs).to(dtype=dtype)[None, None, :, :]
        sin = torch.sin(freqs).to(dtype=dtype)[None, None, :, :]

        x_rope = x[..., : self.dim]
        x_pass = x[..., self.dim :]

        x_even = x_rope[..., 0::2]
        x_odd = x_rope[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)
        if x_pass.numel() == 0:
            return x_rot

        return torch.cat((x_rot, x_pass), dim=-1)
