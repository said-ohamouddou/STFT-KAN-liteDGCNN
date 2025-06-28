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
    'g4': 3,      # gridsize for layer 4 (first linear layer)
    'w4': 25,     # window_size for layer 4
    's4': 14,     # stride for layer 4
    'g5': 4,      # gridsize for layer 5 (second linear layer)
    'w5': 160,    # window_size for layer 5
    's5': 9,      # stride for layer 5
    'sm4': False, # smooth_initialization for layer 4
    'sm5': True,  # smooth_initialization for layer 5
    'wtp4': 'hamming',  # window_type for layer 4
    'wtp5': 'boxcar'    # window_type for layer 5
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

class STFTKanMlpDGCNN(torch.nn.Module):
    """
    Hybrid model combining MLP for edge convolution and STFT-Fourier KAN for feature transformation.
    """
    def __init__(self, out_channels, k, aggr, emb_dims, g4, w4, s4, g5, w5, s5, sm4, sm5, wtp4, wtp5):
        super().__init__()
        
        # Dynamic Edge Convolution with standard MLP
        self.conv = DynamicEdgeConv(MLP([2 * 3, 64, 128],plain_last=False), k, aggr)

        #Linear layers using STFT-Fourier KAN
        self.linear1 = STFTKANLayer(
            128, emb_dims, 
            gridsize=3,           # Number of frequency bands
            window_size=25,       # Size of each STFT window
            stride=28,            # Stride between windows
            smooth_initialization=False,
            window_type='hamming',  # Window function for spectral leakage reduction
            addbias=True
        )
        
        # Output layer with Hann window for better frequency resolution
        self.linear2 = STFTKANLayer(
            emb_dims * 2, out_channels, 
            gridsize=4, 
            window_size=160,
            stride=9,
            smooth_initialization=True,
            window_type='boxcar',
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

    return best_test_acc, test_true, test_pred

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
    """
    experiments = []
    
    # Define parameter variations for ablation study - CORRECTED
    param_variations = {
        # Gridsize parameters (max 10 values)
        'g4': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'g5': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        
        # Window size parameters (max 10 values)
        'w4': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        'w5': [120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
        
        # Stride parameters (max 10 values)
        's4': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        's5': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        
        # Smooth initialization
        'sm4': [True, False],
        'sm5': [True, False],
        
        # Window types
        'wtp4': ['hann', 'hamming', 'bartlett', 'blackman', 'kaiser', 'boxcar'],
        'wtp5': ['hann', 'hamming', 'bartlett', 'blackman', 'kaiser', 'boxcar']
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

def run_ablation_study(epochs=100, save_threshold=0.77):
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
    os.makedirs('saved_models_mlp_kan', exist_ok=True)
    os.makedirs('plots_mlp_kan', exist_ok=True)
    
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
            
            model = STFTKanMlpDGCNN(
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


def analyze_ablation_results(df):
    """
    Analyze ablation study results and identify parameter importance.
    """
    print("\n" + "="*100)
    print("STFT-KAN ABLATION STUDY RESULTS")
    print("="*100)
    
    # Remove failed experiments
    successful_df = df[df['Accuracy'] > 0].copy()
    
    if len(successful_df) == 0:
        print("No successful experiments found!")
        return df
    
    # Get baseline accuracy
    baseline_row = successful_df[successful_df['Parameter_Varied'] == 'baseline']
    if len(baseline_row) > 0:
        baseline_accuracy = baseline_row['Accuracy'].iloc[0]
        print(f"\nBaseline Accuracy: {baseline_accuracy:.4f}")
    else:
        print("\nWarning: Baseline experiment not found!")
        baseline_accuracy = successful_df['Accuracy'].mean()
    
    # Calculate performance change for each parameter
    parameter_impact = {}
    
    print(f"\nPARAMETER IMPACT ANALYSIS:")
    print("-" * 80)
    print(f"{'Parameter':<15} {'Best Value':<15} {'Best Acc':<10} {'Worst Acc':<10} {'Range':<10} {'Avg Change':<12}")
    print("-" * 80)
    
    for param in successful_df['Parameter_Varied'].unique():
        if param == 'baseline':
            continue
            
        param_data = successful_df[successful_df['Parameter_Varied'] == param]
        
        if len(param_data) > 0:
            best_acc = param_data['Accuracy'].max()
            worst_acc = param_data['Accuracy'].min()
            range_acc = best_acc - worst_acc
            avg_change = param_data['Accuracy'].mean() - baseline_accuracy
            
            best_value = param_data.loc[param_data['Accuracy'].idxmax(), 'Parameter_Value']
            
            parameter_impact[param] = {
                'best_value': best_value,
                'best_accuracy': best_acc,
                'worst_accuracy': worst_acc,
                'range': range_acc,
                'avg_change': avg_change
            }
            
            print(f"{param:<15} {str(best_value):<15} {best_acc:<10.4f} {worst_acc:<10.4f} {range_acc:<10.4f} {avg_change:<+12.4f}")
    
    # Sort parameters by importance (range of impact)
    sorted_params = sorted(parameter_impact.items(), 
                          key=lambda x: x[1]['range'], reverse=True)
    
    print(f"\nPARAMETER IMPORTANCE RANKING (by accuracy range):")
    print("-" * 50)
    for i, (param, data) in enumerate(sorted_params[:10], 1):
        print(f"{i:2}. {param:<15} Range: {data['range']:.4f}")
    
    # Find best overall configurations
    print(f"\nTOP 10 CONFIGURATIONS:")
    print("-" * 100)
    top_configs = successful_df.nlargest(10, 'Accuracy')
    display_cols = ['Parameter_Varied', 'Parameter_Value', 'Accuracy', 'Num_Parameters']
    print(top_configs[display_cols].to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stft_kan_ablation_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n✓ Full ablation results saved to: {filename}")
    
    # Create parameter impact summary
    impact_df = pd.DataFrame(parameter_impact).T
    impact_filename = f"parameter_impact_summary_{timestamp}.csv"
    impact_df.to_csv(impact_filename)
    print(f"✓ Parameter impact summary saved to: {impact_filename}")
    
    return df, parameter_impact

def create_ablation_plots(df, parameter_impact):
    """
    Create visualization plots for ablation study results.
    """
    try:
        successful_df = df[df['Accuracy'] > 0].copy()
        
        if len(successful_df) == 0:
            print("No successful experiments to plot!")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        
        # 1. Parameter Impact Bar Plot
        plt.figure(figsize=(12, 8))
        
        params = list(parameter_impact.keys())
        ranges = [parameter_impact[p]['range'] for p in params]
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(params)), ranges)
        plt.xticks(range(len(params)), params, rotation=45, ha='right')
        plt.ylabel('Accuracy Range')
        plt.title('Parameter Importance (Accuracy Range)')
        plt.tight_layout()
        
        # 2. Best vs Worst Accuracy by Parameter
        plt.subplot(2, 2, 2)
        best_accs = [parameter_impact[p]['best_accuracy'] for p in params]
        worst_accs = [parameter_impact[p]['worst_accuracy'] for p in params]
        
        x = range(len(params))
        plt.scatter(x, best_accs, label='Best', alpha=0.7, s=50)
        plt.scatter(x, worst_accs, label='Worst', alpha=0.7, s=50)
        plt.xticks(x, params, rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Best vs Worst Accuracy by Parameter')
        plt.legend()
        
        # 3. Accuracy Distribution
        plt.subplot(2, 2, 3)
        plt.hist(successful_df['Accuracy'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Accuracy Distribution Across All Experiments')
        
        # 4. Parameters vs Accuracy
        plt.subplot(2, 2, 4)
        plt.scatter(successful_df['Num_Parameters'], successful_df['Accuracy'], alpha=0.6)
        plt.xlabel('Number of Parameters')
        plt.ylabel('Accuracy')
        plt.title('Model Size vs Accuracy')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"ablation_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Ablation plots saved to: {plot_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {e}")

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
    print("\n" + "="*50)
    print("ABLATION STUDY COMPLETED!")
    print("="*50)
    
    return results_df, parameter_impact

if __name__ == "__main__":
    results, impact = main()
