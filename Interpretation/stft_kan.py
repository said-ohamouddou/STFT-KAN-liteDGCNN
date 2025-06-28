import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class STFTKANLayer(nn.Module):
    """
    Optimized STFT-based Fourier KAN Layer.

    This layer applies a windowed Fourier transform-like operation to the input.
    Optimizations include precomputed basis functions, reduced tensor operations,
    and more efficient memory usage.
    """
    def __init__(
        self,
        inputdim,
        outdim,
        gridsize,
        window_size,
        stride,
        addbias=True,
        smooth_initialization=False,
        window_type='hann',
        kaiser_beta=14.0,
        device=None
    ):
        """
        Initializes the optimized STFTFourierKANLayer.

        Args:
            inputdim (int): Length of the input.
            outdim (int): Number of output features.
            gridsize (int): Number of frequency bins per window.
            window_size (int): Size of each time window.
            stride (int): Stride between windows.
            addbias (bool): Whether to include a bias term.
            smooth_initialization (bool): If True, attenuates high-frequency coefficients.
            window_type (str): Type of window function ('hann', 'hamming', 'bartlett', 'blackman', 'kaiser','boxcar').
            kaiser_beta (float): Beta parameter for the kaiser window.
            device (torch.device or str): Device to run the layer on.
        """
        super(STFTKANLayer, self).__init__()
        
        # If device not provided, auto-detect
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.inputdim = inputdim
        self.outdim = outdim
        self.gridsize = gridsize
        self.window_size = window_size
        self.stride = stride
        self.addbias = addbias

        # Compute num_windows
        if self.inputdim >= self.window_size:
            self.num_windows = ((self.inputdim - self.window_size) // self.stride) + 1
        else:
            self.num_windows = 1

        # Total length after considering the windows
        self.total_length = (self.num_windows - 1) * self.stride + self.window_size
        
        # Store padding amount for optimization
        self.pad_amount = max(0, self.total_length - self.inputdim)

        # Create window functions
        if window_type == 'hann':
            window = torch.hann_window(self.window_size, device=self.device)
        elif window_type == 'hamming':
            window = torch.hamming_window(self.window_size, device=self.device)
        elif window_type == 'bartlett':
            window = torch.bartlett_window(self.window_size, device=self.device)
        elif window_type == 'blackman':
            window = torch.blackman_window(self.window_size, device=self.device)
        elif window_type == 'kaiser':
            window = torch.kaiser_window(self.window_size, periodic=True, beta=kaiser_beta, device=self.device)
        elif window_type == 'boxcar':
            window = torch.ones(self.window_size, device=self.device)
        else:
            raise ValueError("Unsupported window type. Choose from 'hann', 'hamming', 'bartlett', 'blackman', 'kaiser','boxcar'.")

        # Normalize window to ensure energy preservation
        window = window / window.sum()
        self.register_buffer('window', window)

        # OPTIMIZATION: Precompute basis functions
        n = torch.arange(self.window_size, dtype=torch.float32, device=self.device)
        k = torch.arange(1, self.gridsize + 1, dtype=torch.float32, device=self.device)
        
        # Create meshgrid for n and k
        n_grid, k_grid = torch.meshgrid(n, k, indexing='ij')  # [window_size, gridsize]
        
        # Precompute cos and sin basis functions
        cos_basis = torch.cos(2 * np.pi * k_grid * n_grid / self.window_size)  # [window_size, gridsize]
        sin_basis = torch.sin(2 * np.pi * k_grid * n_grid / self.window_size)  # [window_size, gridsize]
        
        # Register as buffers (non-trainable parameters that move with the model)
        self.register_buffer('cos_basis', cos_basis)
        self.register_buffer('sin_basis', sin_basis)

        # Normalization factor for Fourier coefficients
        if smooth_initialization:
            grid_norm_factor = (torch.arange(1, self.gridsize + 1, dtype=torch.float32, device=self.device) ** 2)
        else:
            grid_norm_factor = torch.full((self.gridsize,), np.sqrt(self.gridsize), dtype=torch.float32, device=self.device)

        # Initialize Fourier coefficients: [2, outdim, num_windows, gridsize]
        # Reorder dimensions for better memory access patterns: [outdim, num_windows, gridsize, 2]
        self.fouriercoeffs = nn.Parameter(
            torch.randn(self.outdim, self.num_windows, self.gridsize, 2, device=self.device) /
            grid_norm_factor.view(1, 1, self.gridsize, 1)
        )

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(self.outdim, device=self.device))

    def forward(self, x):
        """
        Optimized forward pass of the STFTFourierKANLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., inputdim).

        Returns:
            torch.Tensor: Output tensor of shape (..., outdim).
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        original_shape = x.shape
        # Reshape input to [B, inputdim]
        x = x.reshape(-1, self.inputdim)
        B = x.shape[0]

        if self.pad_amount > 0:
            x_padded = F.pad(x, (0, self.pad_amount), mode='constant', value=0)
        else:
            x_padded = x[:, :self.total_length]

        # Extract windows using unfold
        x_unfold = x_padded.unfold(1, self.window_size, self.stride)  # [B, num_windows, window_size]

        # Apply window function efficiently
        x_windowed = x_unfold * self.window.unsqueeze(0).unsqueeze(0)  # [B, num_windows, window_size]

        # x_windowed: [B, num_windows, window_size]
        # cos_basis, sin_basis: [window_size, gridsize]
        cos_projections = torch.einsum('bnw,wg->bng', x_windowed, self.cos_basis)  # [B, num_windows, gridsize]
        sin_projections = torch.einsum('bnw,wg->bng', x_windowed, self.sin_basis)  # [B, num_windows, gridsize]

        # Stack cos and sin projections for efficient coefficient application
        projections = torch.stack([cos_projections, sin_projections], dim=-1)  # [B, num_windows, gridsize, 2]

        # projections: [B, num_windows, gridsize, 2]
        # fouriercoeffs: [outdim, num_windows, gridsize, 2]
        y = torch.einsum('bngc,ongc->bo', projections, self.fouriercoeffs)  # [B, outdim]

        # Add bias if present (in-place operation)
        if self.addbias:
            y = y + self.bias

        # Reshape to original leading dimensions + outdim
        return y.reshape(*original_shape[:-1], self.outdim)

