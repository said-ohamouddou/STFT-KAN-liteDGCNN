"""
@Author: Said Ohamouddou
@File: main.py
@Time: 2025/02/26 13:18 PM
"""
import os 
import argparse
import time
import random
import numpy as np
import wandb
import sklearn.metrics as metrics
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader  import DataLoader
from data import TreePointCloudDataset
from collections import Counter
from models import LiteDGCNN, KanLiteDGCNN, STFTfourierKanLiteDGCNN, STFTfourierKanMLPLiteDGCNN

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
    mean_samples = sum(class_counts.values()) / num_classes
    
    weights = []
    max_count = max(class_counts.values())
    
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        if count < mean_samples:
            # For minority classes, use mean sample count instead
            weight = max_count / mean_samples
        else:
            weight = max_count / count
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train(args):
    # Start tracking total training time
    total_training_start_time = time.time()

    # Initialize Weights & Biases
    run = wandb.init(
        project='Baselines',
        name=args.exp_name,
        reinit=True
    )
  
    # Create datasets with transforms
    train_dataset = TreePointCloudDataset(
        num_points=args.num_points, 
        partition='train'
    )
    test_dataset = TreePointCloudDataset(
        num_points=args.num_points,
        partition='test'
    )
    
    # Create data loaders using standard DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    num_classes = len(train_dataset.classes)
    print(f'Number of classes: {num_classes}')
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # Initialize model based on layer_type
    if args.layer_type == 'mlp':
        model = LiteDGCNN(args, num_classes)
        print("Using MLP in LiteDGCNN model")
    elif args.layer_type == 'stft':
        model = STFTfourierKanLiteDGCNN(args, num_classes)
        print("Using STFT with KAN in LiteDGCNN model")
    elif args.layer_type == 'stftmlp':
        model = STFTfourierKanMLPLiteDGCNN(args, num_classes)
        print("Using MLP with STFT and KAN in LiteDGCNN model")
    else:
        model = KanLiteDGCNN(args, num_classes)
    print(f"Using LiteDGCNN model with '{args.layer_type}' KAN layer type")
    model = model.to(device)
    print(model)
    num_trainable_params = count_trainable_parameters(model)
    print(f"The model has {num_trainable_params:,} trainable parameters.")
    wandb.run.summary["trainable_parameters"] = num_trainable_params
    
   
    print("Use Adam")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-3)
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader, device)
    scaled_weights = class_weights / class_weights.max()
    criterion = torch.nn.CrossEntropyLoss(weight=scaled_weights)
    
    # Initialize best metrics
    best_test_acc = 0.0
    # Initialize metrics at best accuracy
    best_acc_balanced_acc = 0.0
    best_acc_kappa = 0.0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
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
            
            preds = logits.max(dim=1)[1]
            train_loss += loss.item() * batch.num_graphs
            train_total += batch.num_graphs
            
            train_true.append(batch.y.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        scheduler.step()
        
        # Calculate training metrics
        avg_train_loss = train_loss / train_total
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_kappa = metrics.cohen_kappa_score(train_true, train_pred)
        
        print(f"Train Epoch: {epoch} | "
              f"Loss: {avg_train_loss:.6f} | "
              f"Accuracy: {train_acc:.6f} | "
              f"Balanced Accuracy: {train_avg_per_class_acc:.6f} | "
              f"Kappa: {train_kappa:.6f}")
       
        # Testing Phase
   
        model.eval()
        test_loss = 0.0
        test_total = 0
        test_pred = []
        test_true = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                
                preds = logits.max(dim=1)[1]
                test_loss += loss.item() * batch.num_graphs
                test_total += batch.num_graphs
                
                test_true.append(batch.y.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
        
        # Calculate testing metrics
        avg_test_loss = test_loss / test_total
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_kappa = metrics.cohen_kappa_score(test_true, test_pred)
        
        epoch_time = time.time() - epoch_start_time
        current_total_time = time.time() - total_training_start_time
    
        # Log metrics to WandB
        wandb.log({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "test_loss": avg_test_loss,
            "test_acc": test_acc,
            "test_avg_per_class_acc": test_avg_per_class_acc,
            "test_kappa": test_kappa,
            "epoch_time": epoch_time,
            "total_training_time": current_total_time,
            "epoch": epoch
        })
        
        print(f"Test Epoch: {epoch} | "
              f"Loss: {avg_test_loss:.6f} | "
              f"Accuracy: {test_acc:.6f} | "
              f"Balanced Accuracy: {test_avg_per_class_acc:.6f} | "
              f"Kappa: {test_kappa:.6f} | "
              f"Epoch Time: {epoch_time:.2f}s | "
              f"Total Time: {current_total_time/60:.2f}m")
        
        # Create save directory
        save_path = os.path.join('checkpoints', args.exp_name, 'models')
        os.makedirs(save_path, exist_ok=True)
        
        # Save best model based on accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Track balanced accuracy and kappa at best accuracy epoch
            best_acc_balanced_acc = test_avg_per_class_acc
            best_acc_kappa = test_kappa
            
            torch.save(model.state_dict(), os.path.join(save_path, 'best_acc.t7'))
            report = metrics.classification_report(test_true, test_pred, target_names=test_dataset.classes, output_dict=True, zero_division=True)
            # Extract precision for each class and store it for later use
            best_precision_values = {f"best_precision_{label}": metrics['precision'] * 100 
                           for label, metrics in report.items() 
                           if label in test_dataset.classes}
    
            with open(os.path.join(save_path, 'best_acc_epoch.txt'), 'w') as f:
                f.write(f'Best accuracy model saved at epoch {epoch}\n')
                f.write(f'Accuracy: {best_test_acc:.6f}\n')
                f.write(f'Balanced Accuracy: {best_acc_balanced_acc:.6f}\n')
                f.write(f'Kappa: {best_acc_kappa:.6f}')
            print(f"Best accuracy model saved with accuracy: {best_test_acc:.6f}")
            print(f"At best accuracy, balanced accuracy: {best_acc_balanced_acc:.6f}, kappa: {best_acc_kappa:.6f}")
    
    # Calculate and log total training time
    total_training_time = time.time() - total_training_start_time
    
    # Save final results
    print(f'Best test accuracy: {best_test_acc:.4f}')
    print(f'At best accuracy epoch - balanced acc: {best_acc_balanced_acc:.4f}, kappa: {best_acc_kappa:.4f}')
    print(f'Total training time: {total_training_time :.4f} s')
    
    # Add total training time to wandb summary
    wandb.run.summary["best_test_acc"] = best_test_acc*100
    # Add metrics at best accuracy epoch
    wandb.run.summary["balanced_acc_at_best_acc"] = best_acc_balanced_acc*100
    wandb.run.summary["kappa_at_best_acc"] = best_acc_kappa*100
    wandb.run.summary["total_training_time"] = total_training_time
    wandb.run.summary["time_per_epoch"] = total_training_time / args.epochs
    #Log precision values associated with the best model in overall accuracy
    for class_name, value in best_precision_values.items():
        wandb.run.summary[class_name] = value

    
    # Save final metrics summary
    with open(os.path.join(save_path, 'final_results.txt'), 'w') as f:
        f.write(f'Best test accuracy: {best_test_acc:.4f}\n')
        f.write(f'At best accuracy epoch - balanced acc: {best_acc_balanced_acc:.4f}, kappa: {best_acc_kappa:.4f}\n')
        f.write(f'Total training time: {total_training_time:.4f} s\n')
        
    # Save final model
    final_model_path = os.path.join(save_path, 'last.t7')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    run.finish()
    
    
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Classification using LiteDGCNN with Kolmogorov-Arnold Networks (KANs)')
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default='exp', 
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=1, 
                        help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, default='./checkpoints/stft/models/best_acc.t7', 
                        help='Path to pretrained model')
    parser.add_argument('--eval', action='store_true', 
                        help='Evaluate the model instead of training')
                        
    # Training hyperparameters
    
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, 
                        help='Testing batch size')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Initial learning rate')                  
    # Hardware settings
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA training')
                        
    # Model architecture
    parser.add_argument('--num_points', type=int, default=1024, 
                        help='Number of points in point cloud')
    parser.add_argument('--emb_dims', type=int, default=1024, 
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=8, 
                        help='Number of nearest neighbors to use')
    parser.add_argument('--aggr', type=str, default='max', choices=['max', 'mean', 'sum'], 
                        help='Aggregation method (max, mean, sum)')
                        
    # KAN-specific settings
    kan_group = parser.add_argument_group('KAN Settings')
    
    kan_group.add_argument('--layer_type', type=str, default='bernstein',
                        choices=['spline', 'rbf', 'cheby', 'kaln', 'gram', 'relu','fourier','mlp','stft','stftmlp'],
                        help='Type of layer to use for LiteDGCNN')
    args = parser.parse_args()

    print(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch_geometric.seed_everything(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    if not args.eval:
        train(args)
    else:
        test(args)

