import os
import time
import numpy as np
import sklearn.metrics as metrics
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from collections import Counter
from stft_kan import STFTFourierKANLayer
# Local imports (assuming these are your files)
from data import TreePointCloudDataset
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool
import optuna
import torch_geometric
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import random
import os
import joblib
import matplotlib.pyplot as plt
import optuna
import itertools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch_geometric.seed_everything(1)
torch.backends.cudnn.enabled = True

def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_class_weights(train_loader, device):
    """
    Compute class weights based on class distribution in the training dataset.
    Handles PyTorch Geometric data objects.

    Args:
        train_loader: DataLoader containing PyTorch Geometric data objects
        device: torch device to place the resulting weights tensor

    Returns:
        torch.Tensor: Class weights tensor on specified device
    """
    labels = []

    for batch in train_loader:
        try:
            # For PyG data, y is already a tensor
            batch_labels = batch.y

            # Handle both single label and batch of labels
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze()

            # Convert to list and add to labels
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
        # Example weighting approach (may adjust to your preference)
        if count < mean_samples:
            # For minority classes, scale up more
            weight = max_count / mean_samples
        else:
            weight = max_count / count
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train(
    class_weights,
    train_loader,
    test_loader,
    num_classes,
    g2,
    w2,
    s2,
    g3,
    w3,
    s3,
    sm2,
    sm3,
    wtp2,
    wtp3,
    g4=7,
    w4=52,
    s4=20,
    g5=6,
    w5=197,
    s5=10,
    sm4=True,
    sm5=False,
    wtp4='bartlett',
    wtp5='hann',
    epochs=10
):
    """
    Trains and evaluates the STFTfourierKanDGCNN model.

    Args:
        class_weights (torch.Tensor): Precomputed class weights
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Testing data loader
        num_classes (int): Number of classes
        g2, w2, s2, g3, w3, s3: Hyperparameters for first two layers
        g4, w4, s4, g5, w5, s5: Hyperparameters for last two layers
        sm2, sm3, sm4, sm5: Boolean values for smooth initialization
        wtp2, wtp3, wtp4, wtp5: Window types for each layer
        epochs (int): Number of training epochs
    
    Returns:
        best_test_acc (float): Best test accuracy achieved
    """
   
    # Hyperparameters
    lr = 0.001
    k = 8  # Assuming this is the default from args.k
    aggr = 'max'  # Assuming this is the default from args.aggr
    emb_dims = 1024  # Assuming this is the default from args.emb_dims

    # Initialize model
    model = STFTfourierKanDGCNN(
        out_channels=num_classes,
        k=k,
        aggr=aggr,
        emb_dims=emb_dims,
        g2=g2,
        w2=w2,
        s2=s2,
        g3=g3,
        w3=w3,
        s3=s3,
        g4=g4,
        w4=w4,
        s4=s4,
        g5=g5,
        w5=w5,
        s5=s5,
        sm2=sm2,
        sm3=sm3,
        sm4=sm4,
        sm5=sm5,
        wtp2=wtp2,
        wtp3=wtp3,
        wtp4=wtp4,
        wtp5=wtp5
    ).to(device)

    num_trainable_params = count_trainable_parameters(model)
    #print(f"The model has {num_trainable_params:,} trainable parameters.")

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-3)

    # Scale class weights so the largest class has weight = 1.0
    scaled_weights = class_weights / class_weights.max()
    criterion = nn.CrossEntropyLoss(weight=scaled_weights)

    # Track best metrics
    best_test_acc = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()

        ####################
        # Training Phase
        ####################
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_pred = []
        train_true = []

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

            train_true.append(batch.y.cpu().numpy())
            train_pred.append(preds.cpu().numpy())

        scheduler.step()

        # Calculate training metrics
        avg_train_loss = train_loss / train_total
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_bal_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        ####################
        # Testing Phase
        ####################
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
        avg_test_loss = test_loss / test_total
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_bal_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        # Save best model based on accuracy
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            # You could also torch.save(model.state_dict(), "best_model.pth")

    return best_test_acc


class STFTfourierKanDGCNN(torch.nn.Module):
    """
    Dynamic Graph CNN using STFT-Fourier KAN layers throughout the network.
    """
    def __init__(self, out_channels, k, aggr, emb_dims, g2, w2, s2, g3, w3, s3, sm2, sm3, wtp2, wtp3, g4, w4, s4, g5, w5, s5, sm4, sm5, wtp4, wtp5):
        super().__init__()
        self.k = k  # Number of nearest neighbors
        self.aggr = aggr  # Aggregation method   
        self.emb_dims = emb_dims  # Embedding dimensions
        
        # Sequential STFT-Fourier KAN layers for edge feature extraction
        stft_layers = torch.nn.Sequential(
            STFTFourierKANLayer(
                6, 64, 
                gridsize=g2, 
                window_size=w2, 
                stride=s2,
                smooth_initialization=sm2,
                window_type=wtp2,
                addbias=True
            ), 
            STFTFourierKANLayer(
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
        self.linear1 = STFTFourierKANLayer(
            128, emb_dims, 
            gridsize=g4, 
            window_size=w4, 
            stride=s4,
            smooth_initialization=sm4,
            window_type=wtp4,
            addbias=True
        )
        
        self.linear2 = STFTFourierKANLayer(
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


# Window type options
window_types = ['hann', 'hamming', 'bartlett', 'blackman', 'kaiser', 'boxcar']


def objective(trial):
    """
    Objective function for Optuna optimization with descriptive parameter names.
    """
    # Define hyperparameters to optimize with descriptive names
    g2 = trial.suggest_int('Grid Size (Layer 1 in Edge Conv)', 1, 4)
    w2 = trial.suggest_int('Window Size (Layer 1 in Edge Conv)', 2, 4)
    s2 = trial.suggest_int('Stride (Layer 1 in Edge Conv)', 1, 3)
    g3 = trial.suggest_int('Grid Size (Layer 2 in Edge Conv)', 1, 7)
    w3 = trial.suggest_int('Window Size (Layer 2 in Edge Conv)', 10, 64)
    s3 = trial.suggest_int('Stride (Layer 2 in Edge Conv)', 5, 20)
    
    # Additional parameters for layers 4 and 5
    g4 = trial.suggest_int('Grid Size (Feature Expansion Layer)', 5, 10)
    w4 = trial.suggest_int('Window Size (Feature Expansion Layer)', 20, 60)
    s4 = trial.suggest_int('Stride (Feature Expansion Layer)', 10, 25)
    g5 = trial.suggest_int('Grid Size (Classification Layer)', 5, 8)
    w5 = trial.suggest_int('Window Size (Classification Layer)', 150, 250)
    s5 = trial.suggest_int('Stride (Classification Layer)', 8, 15)
    
    # Boolean parameters
    sm2 = trial.suggest_categorical('Smooth Init (Layer 1 in Edge Conv)', [True, False])
    sm3 = trial.suggest_categorical('Smooth Init (Layer 2 in Edge Conv)', [True, False])
    sm4 = trial.suggest_categorical('Smooth Init (Feature Expansion Layer)', [True, False])
    sm5 = trial.suggest_categorical('Smooth Init (Classification Layer)', [True, False])
    
    # Window type parameters
    wtp2 = trial.suggest_categorical('Window Type (Layer 1 in Edge Conv)', window_types)
    wtp3 = trial.suggest_categorical('Window Type (Layer 2 in Edge Conv)', window_types)
    wtp4 = trial.suggest_categorical('Window Type (Feature Expansion Layer)', window_types)
    wtp5 = trial.suggest_categorical('Window Type (Classification Layer)', window_types)
    
    try:
        # The rest of your function remains the same
        test_acc = train(
            class_weights=class_weights,
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            g2=g2, w2=w2, s2=s2,
            g3=g3, w3=w3, s3=s3,
            g4=g4, w4=w4, s4=s4,
            g5=g5, w5=w5, s5=s5,
            sm2=sm2, sm3=sm3, sm4=sm4, sm5=sm5,
            wtp2=wtp2, wtp3=wtp3, wtp4=wtp4, wtp5=wtp5,
            epochs=300
        )
        
        print(f"Trial {trial.number}:")
        print(f"  Parameters: g2={g2}, w2={w2}, s2={s2}, g3={g3}, w3={w3}, s3={s3}, g4={g4}, w4={w4}, s4={s4}, g5={g5}, w5={w5}, s5={s5}")
        print(f"  Smooth init: sm2={sm2}, sm3={sm3}, sm4={sm4}, sm5={sm5}")
        print(f"  Window types: wtp2={wtp2}, wtp3={wtp3}, wtp4={wtp4}, wtp5={wtp5}")
        print(f"  Achieved accuracy: {test_acc:.4f}")
        
        return test_acc
    except Exception as e:
        print(f"Error during training: {e}")
        # Return a poor value to guide Optuna away from this parameter set
        return float('-inf')

def main():
    """
    Main function that handles the optimization workflow:
    1. Loads the datasets and prepares data loaders
    2. Sets up the Optuna study
    3. Runs the optimization
    4. Visualizes and saves results
    """
    global train_loader, test_loader, class_weights, num_classes
    
    print("=" * 80)
    print("STFT-Fourier KAN Model Hyperparameter Optimization")
    print("=" * 80)
    
    # Create output directory for results
    output_dir = "optuna_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Data preparation
    print("\nPreparing datasets and data loaders...")
    try:
        train_dataset = TreePointCloudDataset(num_points=1024, partition='train')
        test_dataset = TreePointCloudDataset(num_points=1024, partition='test')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8, 
            shuffle=True, 
            num_workers=6, 
            pin_memory=True if torch.cuda.is_available() else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=6,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Loaded training dataset with {len(train_dataset)} samples")
        print(f"Loaded testing dataset with {len(test_dataset)} samples")
        
        # Compute class weights
        class_weights = compute_class_weights(train_loader, device)
        
        # Number of classes
        num_classes = len(train_dataset.classes)
        print(f"Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Step 2: Set up the Optuna study
    print("\nSetting up Optuna study...")
    study_name = "stft_fourier_kan_optimization"
    storage_name = f"sqlite:///{os.path.join(output_dir, study_name)}.db"
    
    # Study configuration - pruners help terminate unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=30,
        interval_steps=10
    )
    
    # Create or load study
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner
        )
        print(f"Loaded existing study with {len(study.trials)} previous trials")
        if len(study.trials) > 0:
            print(f"Best previous result: {study.best_value:.4f}")
    except Exception as e:
        print(f"Creating new study: {e}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner
        )
    
    # Add a fixed trial with the initial parameters if study is new
    if len(study.trials) == 0:
        print("Adding initial trial with default parameters...")
        study.enqueue_trial({
	    'Grid Size (Layer 1 in Edge Conv)': 3, 
	    'Window Size (Layer 1 in Edge Conv)': 2, 
	    'Stride (Layer 1 in Edge Conv)': 2,
	    'Grid Size (Layer 2 in Edge Conv)': 1, 
	    'Window Size (Layer 2 in Edge Conv)': 28, 
	    'Stride (Layer 2 in Edge Conv)': 5,
	    'Grid Size (Feature Expansion Layer)': 7, 
	    'Window Size (Feature Expansion Layer)': 52, 
	    'Stride (Feature Expansion Layer)': 20,
	    'Grid Size (Classification Layer)': 6, 
	    'Window Size (Classification Layer)': 197, 
	    'Stride (Classification Layer)': 10,
	    'Smooth Init (Layer 1 in Edge Conv)': True, 
	    'Smooth Init (Layer 2 in Edge Conv)': False, 
	    'Smooth Init (Feature Expansion Layer)': True, 
	    'Smooth Init (Classification Layer)': False,
	    'Window Type (Layer 1 in Edge Conv)': 'boxcar', 
	    'Window Type (Layer 2 in Edge Conv)': 'blackman', 
	    'Window Type (Feature Expansion Layer)': 'bartlett', 
	    'Window Type (Classification Layer)': 'hann'
			})
    
    # Step 3: Run optimization
    n_trials = 1
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    
    try:
        # Callback to save intermediate results
        def save_checkpoint(study, trial):
            if trial.number % 1 == 0:  # Save every 5 trials
                # Save study object
                joblib.dump(study, os.path.join(output_dir, f"study_checkpoint_{trial.number}.pkl"))
                
                # Create and save visualizations
                save_visualizations(study, output_dir, trial.number)
                
        # Run optimization with callback
        study.optimize(objective, n_trials=n_trials, callbacks=[save_checkpoint])
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    except Exception as e:
        print(f"Error during optimization: {e}")
    
    # Step 4: Process results
    print("\nOptimization completed!")
    
    # Print final results
    print("\nBest Trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params:")
    for param, value in trial.params.items():
        print(f"    {param}: {value}")
    
    # Save final visualizations
    save_visualizations(study, output_dir, "final")
    
    # Map descriptive parameter names to short names if needed
    param_mapping = {
        'Grid Size (Layer 1 in Edge Conv)': 'g2',
        'Window Size (Layer 1 in Edge Conv)': 'w2',
        'Stride (Layer 1 in Edge Conv)': 's2',
        'Grid Size (Layer 2 in Edge Conv)': 'g3',
        'Window Size (Layer 2 in Edge Conv)': 'w3',
        'Stride (Layer 2 in Edge Conv)': 's3',
        'Grid Size (Feature Expansion Layer)': 'g4',
        'Window Size (Feature Expansion Layer)': 'w4',
        'Stride (Feature Expansion Layer)': 's4',
        'Grid Size (Classification Layer)': 'g5',
        'Window Size (Classification Layer)': 'w5',
        'Stride (Classification Layer)': 's5',
        'Smooth Init (Layer 1 in Edge Conv)': 'sm2',
        'Smooth Init (Layer 2 in Edge Conv)': 'sm3',
        'Smooth Init (Feature Expansion Layer)': 'sm4',
        'Smooth Init (Classification Layer)': 'sm5',
        'Window Type (Layer 1 in Edge Conv)': 'wtp2',
        'Window Type (Layer 2 in Edge Conv)': 'wtp3',
        'Window Type (Feature Expansion Layer)': 'wtp4',
        'Window Type (Classification Layer)': 'wtp5'
    }
    
    # Convert parameters if needed
    train_params = {}
    for param, value in study.best_params.items():
        # If the parameter uses the descriptive name, map it to the short name
        if param in param_mapping:
            train_params[param_mapping[param]] = value
        else:
            train_params[param] = value
    
    # Train a final model with the best parameters
    """print("\nTraining final model with best parameters...")
    final_accuracy = train(
        class_weights=class_weights,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        **train_params,
        epochs=300  # More epochs for final model
    )"""
    #print(f"Final model accuracy: {final_accuracy:.4f}")
    
    # Save the complete study for later analysis
    joblib.dump(study, os.path.join(output_dir, "final_study.pkl"))
    print(f"\nResults saved to {output_dir}/")

from optuna.importance import FanovaImportanceEvaluator

def save_visualizations(study, output_dir, suffix):
    """
    Creates and saves high-quality Optuna visualizations with scientific paper standards.
    Keeps original parameter names, maintains black color scheme, and removes all titles.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save visualizations
        suffix: String suffix to add to filenames (e.g. trial number or "final")
    """
    try:
        # Set scientific paper quality figure defaults
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        
        # 1. Parameter importance plot with black color scheme
        plt.figure(figsize=(8, 6), facecolor='white')
        
        # Use optuna to plot parameter importances - returns an Axes object
        ax = optuna.visualization.matplotlib.plot_param_importances(study,evaluator=FanovaImportanceEvaluator(seed = 1))
        fig = ax.figure  # Get the figure from the axes
        
        # Remove title
        ax.set_title("")
        
        # Set black color for bars
        for bar in ax.containers:
            for patch in bar:
                patch.set_facecolor('black')
     
        
        # Fix legend color to be black
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_color('black')
            try:
                for handle in legend.legend_handles:
                    handle.set_color('black')
            except AttributeError:
                try:
                    for handle in legend.get_lines():
                        handle.set_color('black')
                except (AttributeError, TypeError):
                    pass
         # Ensure axes labels are black
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.set_xlabel("Importance Score", fontsize=12, color='black')
        ax.set_ylabel("Hyperparameters", fontsize=12, color='black')
        ax.grid(True, axis='x', alpha=0.3, color='lightgray', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"param_importances_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create individual contour plots for parameter pairs
        
        # Get parameter importances
        importances = optuna.importance.get_param_importances(study,evaluator=FanovaImportanceEvaluator(seed = 1))
        print(importances)
        # Get the top 4 important parameters
        top_params = list(importances.keys())[:4]
        if len(top_params) < 2:
            print("Not enough parameters for contour plots")
            return
            
        # Generate all combinations of top parameters (pairs)
        param_pairs = list(itertools.combinations(top_params, 2))
        
        # Create individual contour plots for each pair of parameters
        for param_x, param_y in param_pairs:
            # Create contour plot
            plt.figure(figsize=(8, 6), facecolor='white')
            
            # Get the Axes object
            ax = optuna.visualization.matplotlib.plot_contour(
                study,
                params=[param_x, param_y]
            )
            
            # Remove title as requested
            ax.set_title("")
            
            # Ensure axes labels are black
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            
            # Use a scientific colormap
            for collection in ax.collections:
                if hasattr(collection, 'colorbar'):
                    cbar = collection.colorbar
                    if cbar:
                        cbar.set_label("Objective Value", fontsize=12, color='black')
                        cbar.ax.tick_params(labelcolor='black')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color='lightgray', linestyle='--')
            
            # Format filename: remove special characters and spaces
            param_x_name = param_x.replace(' ', '_').replace('(', '').replace(')', '')
            param_y_name = param_y.replace(' ', '_').replace('(', '').replace(')', '')
            filename = f"contour_{param_x_name}_vs_{param_y_name}_{suffix}.png"
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Add a parallel coordinate plot
        plt.figure(figsize=(10, 6), facecolor='white')
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=top_params)
        
        # Remove title
        ax.set_title("")
        
        # Ensure axes labels are black
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"parallel_coordinate_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Add an optimization history plot
        plt.figure(figsize=(8, 6), facecolor='white')
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        
        # Remove title
        ax.set_title("")
        
        # Ensure axes labels are black
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"optimization_history_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Add a slice plot for the most important parameter
        if top_params:
            plt.figure(figsize=(8, 6), facecolor='white')
            ax = optuna.visualization.matplotlib.plot_slice(study, params=[top_params[0]])
            
            # Remove title
            ax.set_title("")
            
            # Ensure axes labels are black
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"slice_plot_{suffix}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
if __name__ == "__main__":
    # Call main function
    main()
