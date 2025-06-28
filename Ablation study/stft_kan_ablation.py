# STFT-KAN Layer Ablation Study
import os
import random
import time
from collections import Counter
from datetime import datetime
from itertools import product

# Third-party numerical and scientific computing libraries
import numpy as np
import pandas as pd

# Machine learning libraries
import sklearn.metrics as metrics

# PyTorch core
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# PyTorch Geometric
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Custom/local modules 
from data import TreePointCloudDataset
from stft_extra import STFTKANLayer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
torch_geometric.seed_everything(1)
torch.backends.cudnn.enabled = True


# BASELINE CONFIGURATION
BASELINE_CONFIG = {
    'g2': 2,      # gridsize for layer 1
    'w2': 2,      # window_size for layer 1 
    's2': 2,      # stride for layer 1
    'g3': 3,      # gridsize for layer 2
    'w3': 15,     # window_size for layer 2
    's3': 5,      # stride for layer 2
    'g4': 4,      # gridsize for layer 3
    'w4': 25,     # window_size for layer 3
    's4': 14,     # stride for layer 3
    'g5': 4,      # gridsize for layer 4
    'w5': 160,    # window_size for layer 4
    's5': 9,      # stride for layer 4
    'sm2': True,  # smooth_initialization for layer 1
    'sm3': False, # smooth_initialization for layer 1
    'sm4': False, # smooth_initialization for layer 3
    'sm5': True,  # smooth_initialization for layer 4
    'wtp2': 'boxcar',   # window_type for layer 1
    'wtp3': 'bartlett', # window_type for layer 2
    'wtp4': 'hamming',  # window_type for layer 3
    'wtp5': 'boxcar'    # window_type for layer 4
}

def count_trainable_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_class_weights(train_loader, device):
    """Compute class weights based on class distribution in the training dataset."""
    labels = []

    for batch in train_loader:
        try:
            batch_labels = batch.y
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze()
            labels.extend(batch_labels.cpu().numpy().tolist())
        except (IndexError, ValueError, AttributeError) as e:
            print(f"Warning: Skipping a batch due to error: {e}")

    class_counts = Counter(labels)
    num_classes = len(class_counts)
    max_count = max(class_counts.values())
    mean_samples = sum(class_counts.values()) / num_classes

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        if count < mean_samples:
            weight = max_count / mean_samples
        else:
            weight = max_count / count
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32).to(device)

class STFTKanDGCNN(torch.nn.Module):
    """Dynamic Graph CNN using STFT-Fourier KAN layers throughout the network."""
    
    def __init__(self, out_channels, k, aggr, emb_dims, g2, w2, s2, g3, w3, s3, 
                 sm2, sm3, wtp2, wtp3, g4, w4, s4, g5, w5, s5, sm4, sm5, wtp4, wtp5):
        super().__init__()
        self.k = k
        self.aggr = aggr
        self.emb_dims = emb_dims
        
        # Sequential STFT-Fourier KAN layers for edge feature extraction
        stft_layers = torch.nn.Sequential(
            STFTKANLayer(
                6, 64, 
                gridsize=g2, 
                window_size=w2, 
                stride=s2,
                smooth_initialization=sm2,
                window_type=wtp2,
                addbias=True
            ), 
            STFTKANLayer(
                64, 128, 
                gridsize=g3, 
                window_size=w3, 
                stride=s3,
                smooth_initialization=sm3,
                window_type=wtp3,
                addbias=True
            )
        )
        
        # Dynamic Edge Convolution with STFT layers
        self.conv = DynamicEdgeConv(stft_layers, k, aggr)

        # Linear layers using STFT-Fourier KAN
        self.linear1 = STFTKANLayer(
            128, emb_dims, 
            gridsize=g4, 
            window_size=w4, 
            stride=s4,
            smooth_initialization=sm4,
            window_type=wtp4,
            addbias=True
        )
        
        self.linear2 = STFTKANLayer(
            emb_dims * 2, out_channels, 
            gridsize=g5, 
            window_size=w5,
            stride=s5,
            smooth_initialization=sm5,
            window_type=wtp5,
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

def train_single_experiment(model, train_loader, test_loader, class_weights, epochs=100):
    """
    Train a single model configuration and return the best test accuracy.
    """
    lr = 0.001
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-3)

    # Scale class weights so the largest class has weight = 1.0
    scaled_weights = class_weights / class_weights.max()
    criterion = nn.CrossEntropyLoss(weight=scaled_weights)

    best_test_acc = 0.0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_loss += loss.item() * batch.num_graphs
            train_correct += (preds == batch.y).sum().item()
            train_total += batch.num_graphs

        scheduler.step()

        # Testing Phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_pred = []
        test_true = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)

                preds = logits.argmax(dim=1)
                test_loss += loss.item() * batch.num_graphs
                test_correct += (preds == batch.y).sum().item()
                test_total += batch.num_graphs

                test_true.append(batch.y.cpu().numpy())
                test_pred.append(preds.cpu().numpy())

        # Calculate testing metrics
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)

        # Update best accuracy
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_true = test_true
            best_test_pred = test_pred

    return best_test_acc, best_test_true, best_test_pred

def initialize_data_loaders(batch_size=64, num_workers=6):
    """Initialize data loaders once for all experiments."""
    train_dataset = TreePointCloudDataset(num_points=1024, partition='train')
    test_dataset = TreePointCloudDataset(num_points=1024, partition='test')
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Loaded training dataset with {len(train_dataset)} samples")
    print(f"Loaded testing dataset with {len(test_dataset)} samples")
    
    # Determine number of classes from the dataset
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    class_names= train_dataset.classes
    
    return train_loader, test_loader, num_classes, class_names

def generate_ablation_experiments():
    """
    Generate ablation study experiments by varying one parameter at a time.
    For each parameter, generates 2 values below and 4 values above the baseline.
    """
    experiments = []
    
    
    # Define parameter variations for ablation study
    # 2 values below baseline, 4 values above baseline
    param_variations = {
        # Gridsize parameters
        'g2': [1, 3, 4, 5, 6],
        'g3': [1, 2, 4, 5, 6, 7],
        'g4': [2, 3, 5, 6, 7, 8],
        'g5': [2, 3, 5, 6, 7, 8],
        
        # Window size parameters
        'w2': [1, 3, 4, 5, 6],
        'w3': [5, 10, 20, 25, 30, 35],
        'w4': [15, 20, 30, 35, 40, 45],
        'w5': [60, 110, 210, 260, 310, 360],
        
        # Stride parameters
        's2': [1, 4, 6, 8, 10],
        's3': [1, 3, 7, 9, 11, 13],
        's4': [4, 9, 19, 24, 29, 34],
        's5': [4, 14, 19, 24, 29],
        
        # Boolean parameters
        'sm2': [False],
        'sm3': [True],
        'sm4': [True],
        'sm5': [False],
        
        # Window type parameters
        'wtp2': ['hann', 'hamming', 'bartlett', 'blackman', 'kaiser'],
        'wtp3': ['hann', 'hamming', 'boxcar', 'blackman', 'kaiser'],
        'wtp4': ['hann', 'bartlett', 'boxcar', 'blackman', 'kaiser'],
        'wtp5': ['hann', 'hamming', 'bartlett', 'blackman', 'kaiser']
    }
    
    # Add baseline experiment first
    baseline_exp = BASELINE_CONFIG.copy()
    baseline_exp['parameter_varied'] = 'baseline'
    baseline_exp['parameter_value'] = 'baseline'
    experiments.append(baseline_exp)
    # Generate single-parameter ablation experiments
    for param, values in param_variations.items():
        for value in values:
            # Skip if this is the baseline value
            if value == BASELINE_CONFIG[param]:
                continue
                
            # Create experiment config
            config = BASELINE_CONFIG.copy()
            config[param] = value
            config['parameter_varied'] = param
            config['parameter_value'] = value
            experiments.append(config)
    
    print(f"Generated {len(experiments)} ablation experiments")
    return experiments


def run_ablation_study(epochs=100, save_threshold=0.73):
    """
    Run comprehensive ablation study on STFT-KAN hyperparameters.
    Saves models with accuracy >= save_threshold and generates detailed metrics.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
    from sklearn.preprocessing import label_binarize
    import numpy as np
    
    # Create directories for saving results
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data loaders once
    train_loader, test_loader, num_classes, class_names = initialize_data_loaders()
    
    # Compute class weights once
    class_weights = compute_class_weights(train_loader, device)
    
    # Generate ablation experiments
    experiments = generate_ablation_experiments()
    results = []
    best_models = []
    
    print("Starting STFT-KAN Ablation Study...")
    print(f"Total experiments: {len(experiments)}")
    print(f"Using device: {device}")
    print(f"Model save threshold: {save_threshold}")
    
    for i, config in enumerate(experiments):
        print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
        print(f"Parameter: {config['parameter_varied']} = {config['parameter_value']}")
        
        # Show smooth initialization values for clarity
        if config['parameter_varied'] in ['baseline'] or config['parameter_varied'].startswith('sm'):
            print(f"Smooth init: sm2={config['sm2']}, sm3={config['sm3']}, sm4={config['sm4']}, sm5={config['sm5']}")
        
        try:
            start_time = time.time()
            
            # Create model with current configuration
            model_config = {k: v for k, v in config.items() 
                           if k not in ['parameter_varied', 'parameter_value']}
            
            model = STFTKanDGCNN(
                out_channels=num_classes,
                k=8,
                aggr='max',
                emb_dims=1024,
                **model_config
            ).to(device)
            
            # Count parameters
            num_params = count_trainable_parameters(model)
            
            # Train model
            accuracy, test_true, test_pred = train_single_experiment(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                epochs=epochs
            )
            
            training_time = time.time() - start_time
            
            # Calculate precision, recall, and F1-score
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_true, test_pred, average='weighted', zero_division=0
            )
            
            # Store results
            result = {
                'Experiment_ID': i + 1,
                'Parameter_Varied': config['parameter_varied'],
                'Parameter_Value': str(config['parameter_value']),
                'Accuracy': round(accuracy, 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4),
                'Num_Parameters': num_params,
                'Training_Time_Min': round(training_time / 60, 2),
                **model_config  # Add all config parameters
            }
            
            results.append(result)
            
            print(f"✓ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            print(f"  Parameters: {num_params:,} | Time: {training_time/60:.1f}min")
            
            # Save model and generate plots if accuracy meets threshold
            if accuracy >= save_threshold:
                model_name = f"model_exp_{i+1}_{config['parameter_varied']}_{accuracy:.4f}"
                model_path = f"saved_models/{model_name}.pth"
                
                # Save model state dict
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': model_config,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'experiment_id': i + 1,
                    'parameter_varied': config['parameter_varied'],
                    'parameter_value': config['parameter_value']
                }, model_path)
                
                print(f"✓ Model saved: {model_path}")
                
                # Store info for plotting
                best_models.append({
                    'model_name': model_name,
                    'test_true': test_true,
                    'test_pred': test_pred,
                    'accuracy': accuracy,
                    'experiment_id': i + 1,
                    'num_classes': num_classes
                })
                
                # Generate high-quality plots for best models
                generate_model_plots(
                    test_true, test_pred, num_classes,
                    model_name, accuracy, i + 1, class_names
                )
            
            # Clean up memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"✗ Error in experiment {i+1}: {str(e)}")
            # Add failed experiment
            result = {
                'Experiment_ID': i + 1,
                'Parameter_Varied': config.get('parameter_varied', 'unknown'),
                'Parameter_Value': str(config.get('parameter_value', 'unknown')),
                'Accuracy': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1_Score': 0.0,
                'Num_Parameters': 0,
                'Training_Time_Min': 0.0,
                'Error': str(e),
                **{k: v for k, v in config.items() 
                   if k not in ['parameter_varied', 'parameter_value']}
            }
            results.append(result)
    
    # Create summary of best models
    if best_models:
        print(f"\n🎉 Found {len(best_models)} models with accuracy >= {save_threshold}")
        for model_info in best_models:
            print(f"  - {model_info['model_name']}: {model_info['accuracy']:.4f}")
    
    return pd.DataFrame(results)


def generate_model_plots(test_true, test_pred, num_classes, model_name, accuracy, exp_id, class_names=None):
    """
    Generate high-quality ROC curves and confusion matrix for a model.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from sklearn.preprocessing import label_binarize
    import numpy as np
    
    # Set high-quality plot parameters
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Create class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(test_true, test_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                ax=axes[0], cbar_kws={'label': 'Normalized Count'},
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_xlabel('Predicted Label', fontsize=17)
    axes[0].set_ylabel('True Label', fontsize=17)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    
    # 2. ROC Curves (for multi-class)
    if num_classes > 2:
        # Binarize the output for multi-class ROC
        y_test_bin = label_binarize(test_true, classes=range(num_classes))
        y_pred_bin = label_binarize(test_pred, classes=range(num_classes))
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
        for i, color in zip(range(num_classes), colors):
            axes[1].plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Calculate and plot macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        
        macro_auc = auc(all_fpr, mean_tpr)
        axes[1].plot(all_fpr, mean_tpr, 'k--', lw=3,
                    label=f'Macro-avg (AUC = {macro_auc:.3f})')
        
    else:
        # Binary classification ROC
        fpr, tpr, _ = roc_curve(test_true, test_pred)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=3,
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
    
    # Add diagonal line and formatting
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=14)
    axes[1].set_ylabel('True Positive Rate', fontsize=14)
    axes[1].legend(loc="lower right", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save in both PNG and PDF formats
    plot_path_png = f"plots/{model_name}_metrics.png"
    plot_path_pdf = f"plots/{model_name}_metrics.pdf"
    
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(plot_path_pdf, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ High-quality plots saved: {plot_path_png} and {plot_path_pdf}")


def main():
    """
    Main function to run the complete ablation study.
    """
    print("STFT-KAN Hyperparameter Ablation Study")
    print("="*50)
    
    # Display baseline configuration
    print("\nBaseline Configuration:")
    for param, value in BASELINE_CONFIG.items():
        print(f"  {param}: {value}")
    
    # Show what will be tested
    test_experiments = generate_ablation_experiments()
    
    print(f"\nThis study will test:")
    print(f"  - Individual variations of all parameters (one at a time)")
    print(f"  - Total: {len(test_experiments)} experiments")
    
    # Ask user for confirmation
    response = input(f"\nProceed with ablation study? (y/n): ")
    if response.lower() != 'y':
        print("Ablation study cancelled.")
        return
    
    # Run ablation study
    print("\nStarting ablation study...")
    results_df = run_ablation_study(epochs=300)
    
    # Save results to CSV files
    print("\nSaving results...")
    
    # Create timestamp for unique filenames
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_filename = f"ablation_results_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"  - Detailed results saved to: {results_filename}")
    print("\n" + "="*50)
    print("ABLATION STUDY COMPLETED!")
    print("="*50)
    print(f"\nResults saved with timestamp: {timestamp}")
    
    return results_df

if __name__ == "__main__":
    results = main()
