import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import entropy
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def extract_classification_layer_frequencies(model, classes, device):
    """
    Extract frequency patterns from the classification layer for each class.
    """
    if not hasattr(model, 'linear2') or not hasattr(model.linear2, 'fouriercoeffs'):
        return None
    
    layer = model.linear2
    
    with torch.no_grad():
        # Get learned coefficients [outdim, num_windows, gridsize, 2]
        coeffs = layer.fouriercoeffs.cpu().numpy()
        
        # Compute magnitude 
        cos_coeffs = coeffs[:, :, :, 0]
        sin_coeffs = coeffs[:, :, :, 1]
        magnitude = np.sqrt(cos_coeffs**2 + sin_coeffs**2)
        
        # Frequency bins
        freq_bins = np.arange(1, layer.gridsize + 1) / layer.window_size
        
        # For each output class, compute frequency preference
        class_freq_patterns = {}
        
        for class_idx, class_name in enumerate(classes):
            if class_idx < magnitude.shape[0]:  # Ensure we don't exceed output dimensions
                # Average across windows for this class
                class_magnitude = np.mean(magnitude[class_idx, :, :], axis=0)
                
                # Normalize to get frequency distribution
                total_energy = np.sum(class_magnitude)
                freq_distribution = class_magnitude / (total_energy + 1e-10)
                
                # Find dominant frequencies for this class
                dominant_indices = np.argsort(class_magnitude)[-3:][::-1]
                dominant_freqs = freq_bins[dominant_indices]
                dominant_energies = class_magnitude[dominant_indices]
                
                class_freq_patterns[class_name] = {
                    'frequency_distribution': freq_distribution,
                    'dominant_frequencies': dominant_freqs,
                    'dominant_energies': dominant_energies,
                    'total_energy': total_energy
                }
    
    return {
        'freq_bins': freq_bins,
        'class_patterns': class_freq_patterns,
        'layer_info': {
            'inputdim': layer.inputdim,
            'outdim': layer.outdim,
            'window_size': layer.window_size,
            'gridsize': layer.gridsize,
            'num_windows': layer.num_windows
        }
    }

def analyze_learned_frequency_patterns(model, layer_name, device):
    """
    Analyze the actual learned frequency patterns in STFT coefficients.
    """
    layer = None
    
    # Get the specific layer
    if 'conv.nn.' in layer_name:
        idx = int(layer_name.split('.')[-1])
        if hasattr(model, 'conv') and hasattr(model.conv, 'nn'):
            layer = model.conv.nn[idx]
    elif layer_name == 'linear1':
        layer = getattr(model, 'linear1', None)
    elif layer_name == 'linear2':
        layer = getattr(model, 'linear2', None)
    
    if layer is None or not hasattr(layer, 'fouriercoeffs'):
        return None
    
    with torch.no_grad():
        # Get learned coefficients
        coeffs = layer.fouriercoeffs.cpu().numpy()  # [outdim, num_windows, gridsize, 2]
        
        # Compute magnitude
        cos_coeffs = coeffs[:, :, :, 0]
        sin_coeffs = coeffs[:, :, :, 1]
        magnitude = np.sqrt(cos_coeffs**2 + sin_coeffs**2)
        
        # Analyze frequency selectivity
        freq_bins = np.arange(1, layer.gridsize + 1) / layer.window_size
        
        # Average across output dimensions and windows to get frequency preference
        avg_magnitude_per_freq = np.mean(magnitude, axis=(0, 1))
        
        # Compute learned frequency characteristics
        total_energy = np.sum(avg_magnitude_per_freq)
        freq_distribution = avg_magnitude_per_freq / (total_energy + 1e-10)
        
        # Find dominant frequencies
        dominant_freq_idx = np.argsort(avg_magnitude_per_freq)[-5:][::-1]
        dominant_freqs = freq_bins[dominant_freq_idx]
        dominant_energies = avg_magnitude_per_freq[dominant_freq_idx]
        
        # Compute selectivity metrics
        entropy_val = entropy(freq_distribution + 1e-10)
        spectral_centroid = np.sum(freq_bins * freq_distribution)
        
        # Check for learning (compare to random initialization)
        coeff_variance = np.var(coeffs)
        max_magnitude = np.max(magnitude)
        
        return {
            'layer_info': {
                'name': layer_name,
                'inputdim': layer.inputdim,
                'outdim': layer.outdim,
                'window_size': layer.window_size,
                'gridsize': layer.gridsize,
                'num_windows': layer.num_windows
            },
            'frequency_analysis': {
                'freq_bins': freq_bins,
                'magnitude_spectrum': avg_magnitude_per_freq,
                'frequency_distribution': freq_distribution,
                'dominant_frequencies': dominant_freqs,
                'dominant_energies': dominant_energies,
                'spectral_centroid': spectral_centroid,
                'entropy': entropy_val
            },
            'learning_indicators': {
                'coefficient_variance': coeff_variance,
                'max_magnitude': max_magnitude,
                'total_energy': total_energy,
                'is_learned': coeff_variance > 0.01 and max_magnitude > 0.1
            }
        }

def model_interpreter(model, classes, save_dir='interpretation_analysis_before'):
    """
    STFT model interpretation with 3 clean focused visualizations.
    
    Args:
        model: The STFT-based model to interpret
        classes: List of class names
        save_dir: Directory to save visualizations
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    print(f"Starting STFT Model Interpretation...")
    print(f"Results will be saved to: {save_dir}")
    
    # Layer name mapping
    layer_name_mapping = {
        'conv.nn.0': 'ECL',
        'conv.nn.1': 'FEL', 
        'linear1': 'GFA',
        'linear2': 'CL'
    }
    
    # Find STFT layers
    stft_layer_names = []
    
    # Check DynamicEdgeConv layers
    if hasattr(model, 'conv') and hasattr(model.conv, 'nn'):
        for i, layer in enumerate(model.conv.nn):
            if hasattr(layer, 'fouriercoeffs'):
                stft_layer_names.append(f'conv.nn.{i}')
    
    # Check linear layers
    if hasattr(model, 'linear1') and hasattr(model.linear1, 'fouriercoeffs'):
        stft_layer_names.append('linear1')
    if hasattr(model, 'linear2') and hasattr(model.linear2, 'fouriercoeffs'):
        stft_layer_names.append('linear2')
    
    if not stft_layer_names:
        print("No STFT layers found in the model!")
        return None
    
    print(f"Found {len(stft_layer_names)} STFT layers: {stft_layer_names}")
    
    # Analyze learned patterns in each layer
    layer_analyses = {}
    
    for layer_name in stft_layer_names:
        print(f"Analyzing layer: {layer_name}")
        analysis = analyze_learned_frequency_patterns(model, layer_name, device)
        
        if analysis:
            layer_analyses[layer_name] = analysis
    
    if not layer_analyses:
        print("No layers could be analyzed!")
        return None
    
    # Extract classification layer frequency patterns
    print("Analyzing classification layer frequency patterns...")
    class_freq_analysis = extract_classification_layer_frequencies(model, classes, device)
    
    # 1. Learned Frequency Magnitude Spectrum
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (layer_name, analysis) in enumerate(layer_analyses.items()):
        freq_bins = analysis['frequency_analysis']['freq_bins']
        magnitude = analysis['frequency_analysis']['magnitude_spectrum']
        
        display_name = layer_name_mapping.get(layer_name, layer_name)
        plt.plot(freq_bins, magnitude, 'o-', color=colors[i % len(colors)], 
                label=display_name, linewidth=3, markersize=6)
    
    plt.title('Learned Frequency Magnitude Spectrum', fontsize=20, fontweight='bold')
    plt.xlabel('Normalized Frequency', fontsize=16)
    plt.ylabel('Learned Magnitude', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learned_frequency_magnitude_spectrum.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'learned_frequency_magnitude_spectrum.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Classification Layer Frequency Preferences by Classes
    if class_freq_analysis is not None:
        plt.figure(figsize=(14, 8))
        
        # Prepare data for heatmap
        class_names = list(class_freq_analysis['class_patterns'].keys())
        freq_bins = class_freq_analysis['freq_bins']
        
        # Create frequency distribution matrix [classes x frequencies]
        freq_matrix = np.zeros((len(class_names), len(freq_bins)))
        
        for i, class_name in enumerate(class_names):
            class_data = class_freq_analysis['class_patterns'][class_name]
            freq_matrix[i, :] = class_data['frequency_distribution']
        
        # Create heatmap
        sns.heatmap(freq_matrix, 
                    xticklabels=[f'{f:.2f}' for f in freq_bins[::max(1, len(freq_bins)//8)]], 
                    yticklabels=class_names,
                    cmap='viridis', 
                    cbar_kws={'label': 'Frequency Preference'})
        
        plt.title('Classification Layer Frequency Preferences by Classes', fontsize=20, fontweight='bold')
        plt.xlabel('Normalized Frequency Bins', fontsize=16)
        plt.ylabel('Classes', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        # Adjust colorbar
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Frequency Preference', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'classification_frequency_preferences.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'classification_frequency_preferences.pdf'), bbox_inches='tight')
        plt.close()
        
        # 3. Class Frequency Uniqueness
        plt.figure(figsize=(12, 8))
        
        # Calculate how different each class is from others
        class_uniqueness = []
        for i in range(len(class_names)):
            # Calculate average distance from other classes
            distances = []
            for j in range(len(class_names)):
                if i != j:
                    dist = np.linalg.norm(freq_matrix[i] - freq_matrix[j])
                    distances.append(dist)
            class_uniqueness.append(np.mean(distances))
        
        # Create bar plot
        colors_unique = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
        bars = plt.bar(range(len(class_names)), class_uniqueness, 
                      color=colors_unique, alpha=0.8, width=0.6)
        
        plt.title('Class Frequency Uniqueness', fontsize=20, fontweight='bold')
        plt.xlabel('Classes', fontsize=16)
        plt.ylabel('Uniqueness Score', fontsize=16)
        plt.xticks(range(len(class_names)), class_names, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_frequency_uniqueness.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'class_frequency_uniqueness.pdf'), bbox_inches='tight')
        plt.close()
        
        # Print class analysis
        print(f"\nClassification Analysis:")
        for class_name, class_data in class_freq_analysis['class_patterns'].items():
            dom_freqs = class_data['dominant_frequencies']
            print(f"  {class_name}: Top frequencies = {dom_freqs[:3]}")
    
    else:
        print("Could not analyze classification layer")
    
    # Print final summary
    print(f"\nInterpretation complete! Saved visualizations to {save_dir}:")
    print("1. learned_frequency_magnitude_spectrum.png/.pdf")
    print("2. classification_frequency_preferences.png/.pdf") 
    print("3. class_frequency_uniqueness.png/.pdf")
    
    return layer_analyses
