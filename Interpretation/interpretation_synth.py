import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time

# Import the STFT KAN Layer
from stft_kan import STFTKANLayer

def create_challenging_biosignal_data():
    """Create extremely challenging long multivariate biosignal time series data"""
    np.random.seed(42)
    
    n_samples = 10000
    seq_length = 4096  # Very long sequences
    n_classes = 12  # Many classes for difficulty
    
    X = []
    y = []
    
    # Define base parameters for each class with DISTINCT frequency bands for interpretability testing
    class_params = [
        # Low Frequency Classes (0.5-2 Hz) - Baseline/Respiratory-like
        {'base_freq': 0.5, 'harmonics': [2, 3], 'noise_level': 0.25, 'drift_freq': 0.02, 'description': 'Very Low Freq'},
        {'base_freq': 1.2, 'harmonics': [2, 4], 'noise_level': 0.27, 'drift_freq': 0.018, 'description': 'Low Freq'},
        {'base_freq': 1.8, 'harmonics': [2, 3], 'noise_level': 0.26, 'drift_freq': 0.021, 'description': 'Respiratory-like'},
        
        # Mid-Low Frequency Classes (3-8 Hz) - Physiological range
        {'base_freq': 3.5, 'harmonics': [2, 3], 'noise_level': 0.28, 'drift_freq': 0.015, 'description': 'Mid-Low Physio'},
        {'base_freq': 5.2, 'harmonics': [2, 4], 'noise_level': 0.30, 'drift_freq': 0.017, 'description': 'Cardiac-like'},
        {'base_freq': 7.8, 'harmonics': [2, 3], 'noise_level': 0.29, 'drift_freq': 0.019, 'description': 'Neural-like'},
        
        # Mid-High Frequency Classes (10-25 Hz) - Neural/Muscle activity
        {'base_freq': 12.5, 'harmonics': [2, 3], 'noise_level': 0.32, 'drift_freq': 0.012, 'description': 'Alpha-like'},
        {'base_freq': 18.0, 'harmonics': [2, 4], 'noise_level': 0.31, 'drift_freq': 0.014, 'description': 'Beta-like'},
        {'base_freq': 24.5, 'harmonics': [2, 3], 'noise_level': 0.33, 'drift_freq': 0.011, 'description': 'Fast Neural'},
        
        # High Frequency Classes (30-50 Hz) - Muscle/EMG-like
        {'base_freq': 32.0, 'harmonics': [2], 'noise_level': 0.35, 'drift_freq': 0.008, 'description': 'Muscle Low'},
        {'base_freq': 42.5, 'harmonics': [2], 'noise_level': 0.37, 'drift_freq': 0.007, 'description': 'Muscle High'},
        
        # Very High Frequency Class (60+ Hz) - Artifact/Interference
        {'base_freq': 65.0, 'harmonics': [2], 'noise_level': 0.40, 'drift_freq': 0.005, 'description': 'Artifact-like'},
    ]
    
    for i in range(n_samples):
        t = np.linspace(0, 20, seq_length)  # 20 seconds of signal
        class_idx = i % n_classes
        params = class_params[class_idx]
        
        # Base signal with subtle class-specific differences
        base_freq = params['base_freq']
        signal = np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics with class-specific patterns
        for harmonic in params['harmonics']:
            amplitude = 0.3 / harmonic  # Decreasing amplitude
            phase = np.random.random() * 2 * np.pi  # Random phase
            signal += amplitude * np.sin(2 * np.pi * base_freq * harmonic * t + phase)
        
        # Add complex modulations to make classification harder
        
        # 1. Amplitude modulation (class-dependent)
        am_freq = 0.1 + class_idx * 0.005
        am_depth = 0.2 + class_idx * 0.01
        signal *= (1 + am_depth * np.sin(2 * np.pi * am_freq * t))
        
        # 2. Frequency modulation (subtle)
        fm_freq = 0.05 + class_idx * 0.003
        fm_depth = 0.05
        freq_mod = base_freq * (1 + fm_depth * np.sin(2 * np.pi * fm_freq * t))
        phase_mod = 2 * np.pi * np.cumsum(freq_mod) * (t[1] - t[0])
        signal += 0.4 * np.sin(phase_mod)
        
        # 3. Non-stationary components (time-varying patterns)
        # Early phase pattern
        early_mask = t < 5
        signal[early_mask] += 0.3 * np.sin(2 * np.pi * 2.5 * t[early_mask]) * np.exp(-t[early_mask]/2)
        
        # Middle phase pattern  
        middle_mask = (t >= 7) & (t <= 13)
        if np.any(middle_mask):
            t_mid = t[middle_mask] - 7
            signal[middle_mask] += 0.2 * np.sin(2 * np.pi * 1.5 * t_mid) * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t_mid))
        
        # Late phase pattern
        late_mask = t > 15
        if np.any(late_mask):
            t_late = t[late_mask] - 15
            signal[late_mask] += 0.25 * np.sin(2 * np.pi * 0.8 * t_late) * np.exp(-t_late/3)
        
        # 4. Add realistic noise components
        noise_level = params['noise_level']
        
        # Gaussian noise
        signal += noise_level * np.random.randn(seq_length)
        
        # Colored noise (1/f noise)
        freqs = np.fft.fftfreq(seq_length, t[1] - t[0])
        freqs[0] = 1e-10  # Avoid division by zero
        noise_spectrum = np.random.randn(seq_length) + 1j * np.random.randn(seq_length)
        noise_spectrum /= np.sqrt(np.abs(freqs))
        colored_noise = np.real(np.fft.ifft(noise_spectrum))
        signal += 0.1 * colored_noise
        
        # 50/60 Hz powerline interference with harmonics
        powerline_freq = 60
        for h in [1, 2, 3]:
            phase = np.random.random() * 2 * np.pi
            amplitude = 0.08 / h
            signal += amplitude * np.sin(2 * np.pi * powerline_freq * h * t + phase)
        
        # 5. Baseline wander (low-frequency drift)
        drift_freq = params['drift_freq']
        drift_amplitude = 0.15 + np.random.random() * 0.1
        drift_phase = np.random.random() * 2 * np.pi
        signal += drift_amplitude * np.sin(2 * np.pi * drift_freq * t + drift_phase)
        
        # 6. Motion artifacts (sudden jumps and trends)
        n_artifacts = np.random.randint(2, 8)
        for _ in range(n_artifacts):
            artifact_start = np.random.randint(0, seq_length - 200)
            artifact_length = np.random.randint(50, 200)
            artifact_end = min(artifact_start + artifact_length, seq_length)
            
            # Sudden baseline shift
            shift_magnitude = (np.random.random() - 0.5) * 0.5
            signal[artifact_start:artifact_end] += shift_magnitude
            
            # High-frequency burst
            if np.random.random() > 0.5:
                burst_freq = 10 + np.random.random() * 40
                burst_amplitude = 0.3 * np.random.random()
                burst_signal = burst_amplitude * np.sin(2 * np.pi * burst_freq * t[artifact_start:artifact_end])
                signal[artifact_start:artifact_end] += burst_signal
        
        # 7. Intermittent patterns (class-specific transients)
        n_transients = np.random.randint(3, 10)
        for _ in range(n_transients):
            trans_start = np.random.randint(0, seq_length - 100)
            trans_length = np.random.randint(30, 100)
            trans_end = min(trans_start + trans_length, seq_length)
            
            # Class-specific transient pattern
            trans_freq = base_freq * (2 + class_idx * 0.5)
            trans_t = t[trans_start:trans_end] - t[trans_start]
            envelope = np.exp(-trans_t / (trans_length * 0.1))
            transient = 0.4 * envelope * np.sin(2 * np.pi * trans_freq * trans_t)
            signal[trans_start:trans_end] += transient
        
        # 8. Missing data (dropouts)
        n_dropouts = np.random.randint(1, 5)
        for _ in range(n_dropouts):
            dropout_start = np.random.randint(0, seq_length - 50)
            dropout_length = np.random.randint(10, 50)
            dropout_end = min(dropout_start + dropout_length, seq_length)
            signal[dropout_start:dropout_end] = 0
        
        # 9. Make classes even more overlapping
        # Add inter-class contamination
        contamination_strength = 0.4
        other_class = (class_idx + np.random.randint(1, n_classes)) % n_classes
        other_params = class_params[other_class]
        contamination = contamination_strength * np.sin(2 * np.pi * other_params['base_freq'] * t)
        signal += 0.2 * contamination
        
        # 10. Saturation and clipping (realistic sensor limits)
        saturation_level = 3.0 + np.random.random()
        signal = np.clip(signal, -saturation_level, saturation_level)
        
        X.append(signal)
        y.append(class_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Use robust scaling to handle outliers and artifacts
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Created frequency-diverse dataset for interpretability testing:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Classes: {n_classes}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Class frequency distribution:")
    for i, params in enumerate(class_params):
        print(f"    Class {i}: {params['base_freq']:5.1f} Hz ({params['description']})")
    print(f"  Frequency bands covered:")
    print(f"    - Very Low (0.5-2 Hz): Classes 0, 1, 2")
    print(f"    - Physiological (3-8 Hz): Classes 3, 4, 5") 
    print(f"    - Neural/Muscle (10-25 Hz): Classes 6, 7, 8")
    print(f"    - High Muscle (30-50 Hz): Classes 9, 10")
    print(f"    - Artifacts (60+ Hz): Class 11")
    print(f"  Expected: Clear frequency band separation for interpretability analysis")
    
    return X_train, y_train, X_test, y_test

# Biosignal Classifier with single STFT-KAN layer optimized for long sequences
class BiosignalSTFTClassifier(nn.Module):
    def __init__(self, input_dim=4096, num_classes=12, window_size=256, gridsize=64):
        super(BiosignalSTFTClassifier, self).__init__()
        
        # Optimized for long biosignal sequences
        self.stft_kan = STFTKANLayer(
            inputdim=input_dim,
            outdim=num_classes,
            gridsize=gridsize,
            window_size=window_size,
            stride=window_size//4,  # 75% overlap for better frequency resolution
            window_type='blackman',  # Better for biosignal analysis
            smooth_initialization=False  # Allow learning of all frequency components
        )
        
    def forward(self, x):
        # Keep original signal characteristics for frequency analysis
        return self.stft_kan(x)


def visualize_biosignal_frequency_importance(model, class_names=None, save_path=None, save_heatmap_pdf=True):
    """Enhanced visualization for biosignal frequency analysis with medical interpretation
    Fixed to work with the correct STFTKANLayer coefficient structure
    
    Args:
        model: The trained model with STFT-KAN layer
        class_names: List of class names for labeling
        save_path: Path to save the heatmap figure (without extension)
        save_heatmap_pdf: Whether to save heatmap as separate PDF (default: True)
    """
    
    # Get coefficients - CORRECTED VERSION for actual implementation
    with torch.no_grad():
        # Actual structure: fouriercoeffs has shape [outdim, num_windows, gridsize, 2]
        coeffs = model.stft_kan.fouriercoeffs.cpu().numpy()  # [outdim, num_windows, gridsize, 2]
        
        # Extract cosine and sine coefficients
        cos_coeffs = coeffs[:, :, :, 0]  # [outdim, num_windows, gridsize]
        sin_coeffs = coeffs[:, :, :, 1]  # [outdim, num_windows, gridsize]
        
        # Calculate magnitude of complex coefficients
        magnitude = np.sqrt(cos_coeffs**2 + sin_coeffs**2)
        
        # Average across windows for analysis
        avg_magnitude = np.mean(magnitude, axis=1)  # Shape: (num_classes, gridsize)
    
    # Create frequency bins with actual frequency values
    # Assuming 20 second signal with 4096 points -> sampling rate = 204.8 Hz
    sampling_rate = 4096 / 20  # 204.8 Hz
    window_size = model.stft_kan.window_size
    freq_resolution = sampling_rate / window_size
    actual_freqs = np.arange(avg_magnitude.shape[1]) * freq_resolution
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(avg_magnitude.shape[0])]
    
    # Save heatmap as separate PDF if requested
    if save_heatmap_pdf and save_path:
        # Create separate figure for heatmap only
        fig_heatmap, ax_heatmap = plt.subplots(1, 1, figsize=(12, 8))
        
        im = ax_heatmap.imshow(avg_magnitude, aspect='auto', cmap='viridis', origin='lower')
        ax_heatmap.set_title('Frequency Importance Heatmap\n(Biosignal Classification - STFT-KAN)', fontsize=14, fontweight='bold', pad=20)
        ax_heatmap.set_xlabel('Frequency (Hz)', fontsize=12)
        ax_heatmap.set_ylabel('Signal Classes', fontsize=12)
        
        # Set frequency ticks
        freq_ticks = np.linspace(0, len(actual_freqs)-1, 10).astype(int)
        ax_heatmap.set_xticks(freq_ticks)
        ax_heatmap.set_xticklabels([f'{actual_freqs[i]:.1f}' for i in freq_ticks], fontsize=10)
        ax_heatmap.set_yticks(range(len(class_names)))
        ax_heatmap.set_yticklabels(class_names, fontsize=10)
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8, aspect=20)
        cbar.set_label('Coefficient Magnitude', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Improve layout
        plt.tight_layout()
        
        # Save heatmap as PDF
        heatmap_path = save_path if save_path.endswith('.pdf') else f"{save_path}_heatmap.pdf"
        fig_heatmap.savefig(heatmap_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Heatmap saved as PDF: {heatmap_path}")
        
        # Close the heatmap figure to free memory
        plt.close(fig_heatmap)
    
    # Create comprehensive visualization (original full figure)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Full frequency heatmap with actual frequency labels
    im1 = axes[0,0].imshow(avg_magnitude, aspect='auto', cmap='viridis', origin='lower')
    axes[0,0].set_title('Frequency Importance Heatmap\n(Biosignal Classification - STFT-KAN)', fontsize=12)
    axes[0,0].set_xlabel('Frequency (Hz)')
    axes[0,0].set_ylabel('Signal Classes')
    
    # Set frequency ticks
    freq_ticks = np.linspace(0, len(actual_freqs)-1, 8).astype(int)
    axes[0,0].set_xticks(freq_ticks)
    axes[0,0].set_xticklabels([f'{actual_freqs[i]:.1f}' for i in freq_ticks])
    axes[0,0].set_yticks(range(len(class_names)))
    axes[0,0].set_yticklabels(class_names)
    
    cbar1 = plt.colorbar(im1, ax=axes[0,0])
    cbar1.set_label('Coefficient Magnitude')
    
    # 2. Frequency profiles by class
    for i, class_name in enumerate(class_names):
        axes[0,1].plot(actual_freqs, avg_magnitude[i], 
                      label=f'Class {i}', linewidth=2, alpha=0.8)
    
    axes[0,1].set_title('Frequency Response Profiles by Class', fontsize=12)
    axes[0,1].set_xlabel('Frequency (Hz)')
    axes[0,1].set_ylabel('Coefficient Magnitude')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add physiological frequency bands with updated ranges
    axes[0,1].axvspan(0, 2, alpha=0.2, color='red', label='Very Low')
    axes[0,1].axvspan(2, 8, alpha=0.2, color='orange', label='Physiological')
    axes[0,1].axvspan(8, 25, alpha=0.2, color='green', label='Neural/Muscle')
    axes[0,1].axvspan(25, 50, alpha=0.2, color='blue', label='High Muscle')
    axes[0,1].axvspan(50, 70, alpha=0.2, color='purple', label='Artifacts')
    
    # 3. Frequency band analysis
    # Define physiological frequency bands with updated ranges
    bands = {
        'Very Low (0-2 Hz)': (0, 2),
        'Physiological (2-8 Hz)': (2, 8),
        'Neural/Muscle (8-25 Hz)': (8, 25),
        'High Muscle (25-50 Hz)': (25, 50),
        'Artifacts (50-70 Hz)': (50, 70),
        'High Freq Noise (70+ Hz)': (70, actual_freqs.max())
    }
    
    band_importance = {}
    for band_name, (low, high) in bands.items():
        low_idx = np.searchsorted(actual_freqs, low)
        high_idx = np.searchsorted(actual_freqs, high)
        if high_idx > low_idx:
            band_importance[band_name] = np.mean(avg_magnitude[:, low_idx:high_idx], axis=1)
        else:
            band_importance[band_name] = np.zeros(len(class_names))
    
    # Plot band importance - FIXED
    x_pos = np.arange(len(class_names))
    bar_width = 0.12
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']
    
    for i, (band_name, importance) in enumerate(band_importance.items()):
        offset = (i - len(bands)/2) * bar_width
        # Ensure importance has the correct shape
        if len(importance) == len(class_names):
            axes[1,0].bar(x_pos + offset, importance, bar_width, 
                         label=band_name, alpha=0.8, color=colors[i % len(colors)])
        else:
            print(f"Warning: Skipping band {band_name} - shape mismatch: {importance.shape}")
    
    axes[1,0].set_title('Frequency Band Importance by Class', fontsize=12)
    axes[1,0].set_xlabel('Signal Classes')
    axes[1,0].set_ylabel('Average Coefficient Magnitude')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels([f'Class {i}' for i in range(len(class_names))])
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Peak frequency analysis
    peak_frequencies = []
    peak_magnitudes = []
    for i in range(len(class_names)):
        peak_idx = np.argmax(avg_magnitude[i])
        peak_freq = actual_freqs[peak_idx]
        peak_mag = avg_magnitude[i][peak_idx]
        peak_frequencies.append(peak_freq)
        peak_magnitudes.append(peak_mag)
    
    scatter = axes[1,1].scatter(peak_frequencies, peak_magnitudes, 
                               c=range(len(class_names)), cmap='tab10', s=100, alpha=0.8)
    
    for i, (freq, mag) in enumerate(zip(peak_frequencies, peak_magnitudes)):
        axes[1,1].annotate(f'C{i}', (freq, mag), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    axes[1,1].set_title('Peak Frequency vs Magnitude by Class', fontsize=12)
    axes[1,1].set_xlabel('Peak Frequency (Hz)')
    axes[1,1].set_ylabel('Peak Magnitude')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print coefficient structure information
    print(f"\nSTFT-KAN Layer Coefficient Structure:")
    print(f"  Fourier coefficients shape: {coeffs.shape}")
    print(f"  Cosine coefficients shape: {cos_coeffs.shape}")
    print(f"  Sine coefficients shape: {sin_coeffs.shape}")
    print(f"  Combined magnitude shape: {magnitude.shape}")
    print(f"  Window-averaged magnitude shape: {avg_magnitude.shape}")
    
    # Return detailed analysis
    return {
        'frequency_magnitude': avg_magnitude,
        'actual_frequencies': actual_freqs,
        'band_importance': band_importance,
        'peak_frequencies': peak_frequencies,
        'peak_magnitudes': peak_magnitudes,
        'sampling_rate': sampling_rate,
        'cos_coefficients': cos_coeffs,
        'sin_coefficients': sin_coeffs
    }


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.linear(x)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.linear(x)

def train_biosignal_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load challenging biosignal data
    X_train, y_train, X_test, y_test = create_challenging_biosignal_data()
    
    # Results storage
    results = []
    
    # ===== TRAIN STFT-KAN MODEL (gridsize=64) ===== 
    print("\n" + "="*50)
    print("TRAINING STFT-KAN MODEL (gridsize=64)")
    print("="*50)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # STFT-KAN Model
    stft_model = BiosignalSTFTClassifier(
        input_dim=4096, 
        num_classes=12,
        window_size=256,
        gridsize=64
    ).to(device)
    
    stft_param_count = sum(p.numel() for p in stft_model.parameters() if p.requires_grad)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(stft_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    print(f"STFT-KAN parameters: {stft_param_count}")
    
    # Training STFT-KAN
    start_time = time.time()
    stft_model.train()
    best_stft_acc = 0
    
    for epoch in range(30):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = stft_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Test evaluation
        stft_model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = stft_model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        
        test_acc = 100. * test_correct / len(test_dataset)
        best_stft_acc = max(best_stft_acc, test_acc)
        
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        stft_model.train()
    
    stft_training_time = time.time() - start_time
    
    # ===== TRAIN STFT-KAN MODEL (gridsize=15) ===== 
    print("\n" + "="*50)
    print("TRAINING STFT-KAN MODEL (gridsize=15)")
    print("="*50)
    
    # STFT-KAN Model with smaller gridsize
    stft_model_small = BiosignalSTFTClassifier(
        input_dim=4096, 
        num_classes=12,
        window_size=256,
        gridsize=15  # Smaller gridsize
    ).to(device)
    
    stft_small_param_count = sum(p.numel() for p in stft_model_small.parameters() if p.requires_grad)
    
    criterion_small = nn.CrossEntropyLoss()
    optimizer_small = optim.AdamW(stft_model_small.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_small = optim.lr_scheduler.CosineAnnealingLR(optimizer_small, T_max=30)
    
    print(f"STFT-KAN (gridsize=15) parameters: {stft_small_param_count}")
    
    # Training STFT-KAN (gridsize=15)
    start_time = time.time()
    stft_model_small.train()
    best_stft_small_acc = 0
    
    for epoch in range(30):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer_small.zero_grad()
            output = stft_model_small(data)
            loss = criterion_small(output, target)
            loss.backward()
            optimizer_small.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler_small.step()
        
        # Test evaluation
        stft_model_small.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = stft_model_small(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        
        test_acc = 100. * test_correct / len(test_dataset)
        best_stft_small_acc = max(best_stft_small_acc, test_acc)
        
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        stft_model_small.train()
    
    stft_small_training_time = time.time() - start_time
    
    # ===== TRAIN LINEAR MODEL ===== 
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*50)
    
    # Logistic Regression Model
    linear_model = LogisticRegression(input_dim=4096, num_classes=12).to(device)
    linear_param_count = sum(p.numel() for p in linear_model.parameters() if p.requires_grad)
    
    criterion_linear = nn.CrossEntropyLoss()
    optimizer_linear = optim.AdamW(linear_model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler_linear = optim.lr_scheduler.CosineAnnealingLR(optimizer_linear, T_max=30)
    
    print(f"Linear model parameters: {linear_param_count}")
    
    # Training Linear
    start_time = time.time()
    linear_model.train()
    best_linear_acc = 0
    
    for epoch in range(30):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer_linear.zero_grad()
            output = linear_model(data)
            loss = criterion_linear(output, target)
            loss.backward()
            optimizer_linear.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        scheduler_linear.step()
        
        # Test evaluation
        linear_model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = linear_model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        
        test_acc = 100. * test_correct / len(test_dataset)
        best_linear_acc = max(best_linear_acc, test_acc)
        
        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        linear_model.train()
    
    linear_training_time = time.time() - start_time
    
    # ===== CREATE RESULTS DATAFRAME =====
    results_data = {
        'Model': ['STFT-KAN (gridsize=64)', 'STFT-KAN (gridsize=15)', 'Logistic Regression'],
        'Best_Test_Accuracy': [best_stft_acc, best_stft_small_acc, best_linear_acc],
        'Training_Time_seconds': [stft_training_time, stft_small_training_time, linear_training_time],
        'Parameter_Count': [stft_param_count, stft_small_param_count, linear_param_count],
        'Training_Speed_samples_per_sec': [
            len(train_dataset) * 30 / stft_training_time,
            len(train_dataset) * 30 / stft_small_training_time,
            len(train_dataset) * 30 / linear_training_time
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    results_df.to_csv('biosignal_model_comparison.csv', index=False)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to 'biosignal_model_comparison.csv'")
    
    return  stft_model, results_df


if __name__ == "__main__":
    # Train the model
    print("Training biosignal classifier with STFT-KAN layer...")
    model , _= train_biosignal_model()
    
    # Visualize frequency importance
    print("\nGenerating biosignal frequency importance visualization...")
    class_names = [f'Class {i}' for i in range(12)]
    analysis = visualize_biosignal_frequency_importance(
        model, 
        class_names=class_names, 
        save_path='biosignal_frequency_analysis.png'
    )
    
    # Enhanced insights for biosignal interpretation
    print("\nBiosignal Frequency Analysis (STFT-KAN):")
    print("="*60)
    
    print(f"\nSampling Rate: {analysis['sampling_rate']:.1f} Hz")
    print(f"Frequency Resolution: {analysis['actual_frequencies'][1]:.2f} Hz per bin")
    print(f"Frequency Range: 0 - {analysis['actual_frequencies'][-1]:.1f} Hz")
    print(f"Number of frequency bins: {len(analysis['actual_frequencies'])}")
    
    # Show peak frequencies for each class
    print(f"\nPeak Frequencies by Class:")
    for i, (freq, mag) in enumerate(zip(analysis['peak_frequencies'], analysis['peak_magnitudes'])):
        print(f"  Class {i}: {freq:.2f} Hz (magnitude: {mag:.3f})")
    
    # Show frequency band importance
    print(f"\nFrequency Band Analysis:")
    for band_name, importance in analysis['band_importance'].items():
        avg_importance = np.mean(importance)
        print(f"  {band_name}: Average importance = {avg_importance:.3f}")
    
    print(f"\nInterpretation Notes:")
    print(f"- Higher coefficient magnitudes indicate greater importance for classification")
    print(f"- Each class should show distinct frequency signatures")
    print(f"- Physiological bands should align with expected biosignal characteristics")
    print(f"- Peak frequencies reveal the most discriminative features per class")
