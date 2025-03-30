import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STFTKANLayer(nn.Module):
    """
    STFT-based Fourier KAN Layer.

    This layer applies a windowed Fourier transform-like operation to the input.
    The user only needs to define stride and window size; 
    the number of windows is concluded from the input dimension.
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
        Initializes the STFTFourierKANLayer.

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
            window = torch.kaiser_window(self.window_size,periodic=True,beta=kaiser_beta,device=self.device)
        elif window_type == 'boxcar':
            window = torch.ones(self.window_size, device=self.device)
        else:
            raise ValueError("Unsupported window type. Choose from 'hann', 'hamming', 'bartlett', 'blackman', 'kaiser','boxcar'.")

        # Normalize window to ensure energy preservation
        window = window / window.sum()
        self.register_buffer('window', window)  # This is stored on device

        # Normalization factor for Fourier coefficients
        if smooth_initialization:
            grid_norm_factor = (torch.arange(1, self.gridsize + 1, dtype=torch.float32, device=self.device) ** 2)  # [G]
        else:
            grid_norm_factor = torch.full((self.gridsize,), np.sqrt(self.gridsize), dtype=torch.float32, device=self.device)  # [G]

        # Initialize Fourier coefficients: [2, outdim, num_windows, gridsize]
        # 2 for cosine and sine
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, self.outdim, self.num_windows, self.gridsize, device=self.device) /
            (grid_norm_factor.view(1, 1, self.gridsize))
        )

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, self.outdim, device=self.device))

    def forward(self, x):
        """
        Forward pass of the STFTFourierKANLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., inputdim).

        Returns:
            torch.Tensor: Output tensor of shape (..., outdim).
        """
        x = x.to(self.device)
        ##print("Input x shape:", x.shape)
        original_shape = x.shape
        # Reshape input to [B, inputdim]
        x = x.reshape(-1, self.inputdim)  # [B, inputdim]
        ##print("Reshaped x to [B, inputdim]:", x.shape)

        B = x.shape[0]
        ##print("Batch size B:", B)
        ##print("Number of windows:", self.num_windows)
        #print("Total length (for windowing):", self.total_length)

        # If input is shorter, pad with zeros
        if self.inputdim < self.total_length:
            pad_amount = self.total_length - self.inputdim
            x_padded = torch.nn.functional.pad(x, (0, pad_amount), mode='constant', value=0)
        else:
            x_padded = x[:, :self.total_length]
        #print("x_padded shape:", x_padded.shape)

        # Extract windows
        x_unfold = x_padded.unfold(1, self.window_size, self.stride)  # [B, num_windows, window_size]
        #print("x_unfold shape:", x_unfold.shape)

        # Apply window function
        window = self.window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size]
        #print("window shape:", window.shape)
        x_windowed = x_unfold * window  # [B, num_windows, window_size]
        #print("x_windowed shape:", x_windowed.shape)

        # Define time (n) and frequency (k) indices
        n = torch.arange(self.window_size, device=self.device).float()  # [window_size]
        n = n.view(1, 1, self.window_size, 1)  # [1,1,window_size,1]
        #print("n shape:", n.shape)

        k = torch.arange(1, self.gridsize + 1, device=self.device).float()
        k = k.view(1, 1, 1, self.gridsize)  # [1,1,1,gridsize]
        #print("k shape:", k.shape)

        # Compute basis functions
        c = torch.cos(2 * np.pi * k * n / self.window_size)  # [1,1,window_size,gridsize]
        s = torch.sin(2 * np.pi * k * n / self.window_size)
        #print("c shape:", c.shape)
        #print("s shape:", s.shape)

        # Prepare for multiplication
        x_windowed = x_windowed.unsqueeze(-1)  # [B,num_windows,window_size,1]
        #print("x_windowed unsqueezed shape:", x_windowed.shape)

        c = c.expand(B, self.num_windows, self.window_size, self.gridsize)  # [B,num_windows,window_size,gridsize]
        s = s.expand(B, self.num_windows, self.window_size, self.gridsize)
        #print("c expanded shape:", c.shape)
        #print("s expanded shape:", s.shape)

        # Multiply to get cos and sin projections
        y_cos_components = x_windowed * c  # [B,num_windows,window_size,gridsize]
        y_sin_components = x_windowed * s

        # Sum over window_size
        y_cos_sum = y_cos_components.sum(dim=2)  # [B, num_windows, gridsize]
        y_sin_sum = y_sin_components.sum(dim=2)  # [B, num_windows, gridsize]

        # Apply Fourier coefficients
        a = self.fouriercoeffs[0]  # [outdim, num_windows, gridsize]
        b = self.fouriercoeffs[1]  # [outdim, num_windows, gridsize]

        a = a.unsqueeze(0)  # [1,outdim,num_windows,gridsize]
        b = b.unsqueeze(0)  # [1,outdim,num_windows,gridsize]

        y_cos_sum = y_cos_sum.unsqueeze(1)  # [B,1,num_windows,gridsize]
        y_sin_sum = y_sin_sum.unsqueeze(1)  # [B,1,num_windows,gridsize]

        Y_cos = (y_cos_sum * a).sum(dim=(2,3))  # [B,outdim]
        Y_sin = (y_sin_sum * b).sum(dim=(2,3))  # [B,outdim]

        y = Y_cos + Y_sin  # [B,outdim]

        if self.addbias:
            y = y + self.bias  # [B,outdim]

        # Reshape to original leading dimensions + outdim
        y = y.reshape(*original_shape[:-1], self.outdim)
        #print("Final output shape:", y.shape)
        return y


