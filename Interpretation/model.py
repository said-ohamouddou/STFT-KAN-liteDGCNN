"""
@Author: Said Ohamouddou
@File: models.py
@Time: 2025/02/26 13:18 PM
"""

import torch
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool
from torch.nn import Linear
from stft_kan import STFTKANLayer

class STFTKanLiteDGCNN(torch.nn.Module):
    """
    Dynamic Graph CNN using STFT-Fourier KAN layers throughout the network.
    """
    def __init__(self, args, out_channels=7):
        super().__init__()
        k = args.k  # Number of nearest neighbors
        aggr = args.aggr  # Aggregation method   
        emb_dims = args.emb_dims  # Embedding dimensions
        
        # Sequential STFT-Fourier KAN layers for edge feature extraction
        stft_layers = torch.nn.Sequential(
            STFTKANLayer(
                6, 64, 
                gridsize=2, 
                window_size=2, 
                stride=2,
                smooth_initialization=True,
                window_type='boxcar',
                addbias=True
            ), 
            STFTKANLayer(
                64, 128, 
                gridsize=3, 
                window_size=30, 
                stride=5,
                smooth_initialization=False,
                window_type='bartlett',
                addbias=True
            )
        )
        
        # Dynamic Edge Convolution with STFT layers
        self.conv = DynamicEdgeConv(stft_layers, k, aggr)

        # Linear layers using STFT-Fourier KAN
        self.linear1 = STFTKANLayer(
            128, args.emb_dims, 
            gridsize=4, 
            window_size=25, 
            stride=14,
            smooth_initialization=False,
            window_type='hamming',
            addbias=True
        )
        
        self.linear2 = STFTKANLayer(
            args.emb_dims * 2, out_channels, 
            gridsize=4, 
            window_size=160,
            stride=9,
            smooth_initialization=True,
            window_type='boxcar',
            addbias=True
        )

    def forward(self, data): 
        """Forward pass through the full STFT-Fourier KAN network"""
        pos, batch = data.pos.float(), data.batch

        # Apply dynamic edge convolution with STFT layers
        x1 = self.conv(pos, batch)

        # Apply first linear STFT layer
        x = self.linear1(x1)

        # Global pooling
        x1 = global_max_pool(x, batch)
        x2 = global_mean_pool(x, batch)

        # Concatenate pooled features
        x = torch.cat((x1, x2), dim=1)

        # Apply final STFT layer
        x = self.linear2(x)

        return x
        
