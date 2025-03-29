"""
@Author: Said Ohamouddou
@File: models.py
@Time: 2025/02/26 13:18 PM
"""

import torch
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool
from torch.nn import Linear
from kans import (
    NaiveFourierKANLayer, KANLayer, FastKANLayer,
    KALNLayer, GRAMLayer, ReLUKANLayer, JacobiKANLayer
)

from stft_kan import STFTFourierKANLayer

def get_kan_layer(in_dim, out_dim, layer_type):
    """
    Create and return the specified KAN layer type with the given input and output dimensions.
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        layer_type (str): Type of KAN layer to use
        
    Returns:
        nn.Module: The appropriate KAN layer instance
    """
    layer_type = layer_type.lower()
    
    if layer_type == 'spline':
        return KANLayer(in_dim, out_dim, grid_size=1, spline_order=0 )
    elif layer_type == 'rbf':
        return FastKANLayer(in_dim, out_dim, grid_min=0, grid_max=1, num_grids=2, use_base_update=False)
    elif layer_type == 'cheby':
        return JacobiKANLayer(in_dim, out_dim, degree=0, a=1/2, b=1/2)
    elif layer_type == 'kaln':
        return KALNLayer(in_dim, out_dim, degree=0)
    elif layer_type == 'gram':
        return GRAMLayer(in_dim, out_dim, degree=0)
    elif layer_type == 'fourier':
        return NaiveFourierKANLayer(in_dim, out_dim, gridsize=1, addbias=True, smooth_initialization=False)
    elif layer_type == 'relu':
        return ReLUKANLayer(in_dim, out_dim, g=1, k=0, train_ab=False )
    else:
        # Default to standard KANLayer if type not recognized
        print(f"Warning: Unknown layer type '{layer_type}'. Using default KANLayer.")
        return KANLayer(in_dim, out_dim)



class MultiLayerKAN(torch.nn.Module):
    """
    Multi-layer KAN network with configurable layer sizes.
    """
    def __init__(self, layer_sizes, args):
        super(MultiLayerKAN, self).__init__()  # Fixed inheritance call
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(get_kan_layer(layer_sizes[i], layer_sizes[i+1], args))
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through sequential KAN layers"""
        return self.layers(x)

# ===================================
# MODEL ARCHITECTURES
# ===================================

class KanLiteDGCNN(torch.nn.Module):
    """
    Dynamic Graph CNN with KAN layers for point cloud classification.
    """
    def __init__(self, args, out_channels=7):
        super().__init__()
        k = args.k  # Number of nearest neighbors
        aggr = args.aggr  # Aggregation method   
        emb_dims = args.emb_dims  # Embedding dimensions
     
        # Dynamic Edge Convolution layer with a multi-layer KAN
        self.conv = DynamicEdgeConv(MultiLayerKAN([2 * 3, 64, 128], args.layer_type), k, aggr)

        # Linear layers using specified KAN layer type
        self.linear1 = get_kan_layer(128, emb_dims, args.layer_type)
        self.linear2 = get_kan_layer(2 * emb_dims, out_channels, args.layer_type)

    def forward(self, data):
        """
        Forward pass through the network.
        """
        pos, batch = data.pos.float(), data.batch

        # Apply dynamic edge convolution to extract local features
        x1 = self.conv(pos, batch)

        # Apply first linear layer to transform features
        x = self.linear1(x1)

        # Global pooling to aggregate features across points
        x1 = global_max_pool(x, batch)  # Max pooling captures most prominent features
        x2 = global_mean_pool(x, batch)  # Mean pooling captures average features

        # Concatenate pooled features for richer representation
        x = torch.cat((x1, x2), dim=1)

        # Apply final linear layer for classification
        x = self.linear2(x)

        return x

class LiteDGCNN(torch.nn.Module):
    """
    Dynamic Graph CNN with standard MLP layers for point cloud classification.
    Serves as a baseline comparison model.
    """
    def __init__(self, args, out_channels=7):
        super().__init__()
        k = args.k  # Number of nearest neighbors
        aggr = args.aggr  # Aggregation method   
        emb_dims = args.emb_dims  # Embedding dimensions
        
        # Dynamic Edge Convolution layer with MLP
        self.conv = DynamicEdgeConv(MLP([2 * 3, 64, 128], plain_last=False), k, aggr)
        
        # Linear layers using MLP
        self.linear1 = MLP([128, emb_dims], plain_last=False)
        self.linear2 = MLP([2 * emb_dims, out_channels])
        
    def forward(self, data):
        """Forward pass through the MLP-based network"""
        pos, batch = data.pos.float(), data.batch
        
        # Apply dynamic edge convolution
        x1 = self.conv(pos, batch)
        
        # Apply first linear layer
        x = self.linear1(x1)
        
        # Global pooling
        x1 = global_max_pool(x, batch)
        x2 = global_mean_pool(x, batch)
        
        # Concatenate pooled features
        x = torch.cat((x1, x2), dim=1)
        
        # Apply final linear layer
        x = self.linear2(x)
        
        return x

class STFTfourierKanMLPLiteDGCNN(torch.nn.Module):
    """
    Hybrid model combining MLP for edge convolution and STFT-Fourier KAN for feature transformation.
    """
    def __init__(self, args, out_channels=7):
        super().__init__()
        k = args.k  # Number of nearest neighbors
        aggr = args.aggr  # Aggregation method   
        emb_dims = args.emb_dims  # Embedding dimensions
        
        # Dynamic Edge Convolution with standard MLP
        self.conv = DynamicEdgeConv(MLP([2 * 3, 64, 128]), k, aggr)

        #Linear layers using STFT-Fourier KAN
        self.linear1 = STFTFourierKANLayer(
            128, args.emb_dims, 
            gridsize=7,           # Number of frequency bands
            window_size=52,       # Size of each STFT window
            stride=20,            # Stride between windows
            smooth_initialization=True,
            window_type='bartlett',  # Window function for spectral leakage reduction
            addbias=True
        )
        
        # Output layer with Hann window for better frequency resolution
        self.linear2 = STFTFourierKANLayer(
            args.emb_dims * 2, out_channels, 
            gridsize=6, 
            window_size=197,
            stride=7,
            smooth_initialization=False,
            window_type='hann',
            addbias=True
        )

    def forward(self, data): 
        """Forward pass through the hybrid network"""
        pos, batch = data.pos.float(), data.batch

        # Apply dynamic edge convolution with MLP
        x1 = self.conv(pos, batch)

        # Apply first STFT-Fourier KAN layer
        x = self.linear1(x1)

        # Global pooling
        x1 = global_max_pool(x, batch)
        x2 = global_mean_pool(x, batch)

        # Concatenate pooled features
        x = torch.cat((x1, x2), dim=1)

        # Apply final STFT-Fourier KAN layer
        x = self.linear2(x)

        return x

class STFTfourierKanLiteDGCNN(torch.nn.Module):
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
            STFTFourierKANLayer(
                6, 64, 
                gridsize=3, 
                window_size=2, 
                stride=2,
                smooth_initialization=True,
                window_type='boxcar',
                addbias=True
            ), 
            STFTFourierKANLayer(
                64, 128, 
                gridsize=1, 
                window_size=28, 
                stride=5,
                smooth_initialization=False,
                window_type='blackman',
                addbias=True
            )
        )
        
        # Dynamic Edge Convolution with STFT layers
        self.conv = DynamicEdgeConv(stft_layers, k, aggr)

        # Linear layers using STFT-Fourier KAN
        self.linear1 = STFTFourierKANLayer(
            128, args.emb_dims, 
            gridsize=7, 
            window_size=52, 
            stride=20,
            smooth_initialization=True,
            window_type='bartlett',
            addbias=True
        )
        
        self.linear2 = STFTFourierKANLayer(
            args.emb_dims * 2, out_channels, 
            gridsize=6, 
            window_size=197,
            stride=10,
            smooth_initialization=False,
            window_type='hann',
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
        
