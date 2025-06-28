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
from model import STFTKanLiteDGCNN
from interpreter_text_export_latex import model_interpreter
import copy
  
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
        project='Final Run',
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
    

    model = STFTKanLiteDGCNN(args, num_classes)
    model = model.to(device)
    print(model)
    results = model_interpreter(model,train_dataset.classes , save_dir='interpretation_analysis_before')
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
    # Store best model predictions for plotting
    best_test_true = None
    best_test_pred = None
    
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
        
        print(f"Train Epoch: {epoch} | "
              f"Loss: {avg_train_loss:.6f} | "
              f"Accuracy: {train_acc:.6f} | "
              f"Balanced Accuracy: {train_avg_per_class_acc:.6f}")
       
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
        
        epoch_time = time.time() - epoch_start_time
        current_total_time = time.time() - total_training_start_time
    
        # Log metrics to WandB
        wandb.log({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "test_loss": avg_test_loss,
            "test_acc": test_acc,
            "test_avg_per_class_acc": test_avg_per_class_acc,
            "epoch_time": epoch_time,
            "total_training_time": current_total_time,
            "epoch": epoch
        })
        
        print(f"Test Epoch: {epoch} | "
              f"Loss: {avg_test_loss:.6f} | "
              f"Accuracy: {test_acc:.6f} | "
              f"Balanced Accuracy: {test_avg_per_class_acc:.6f} | "
              f"Epoch Time: {epoch_time:.2f}s | "
              f"Total Time: {current_total_time/60:.2f}m")
        
        # Create save directory
        save_path = os.path.join('checkpoints_interprter', args.exp_name, 'models')
        os.makedirs(save_path, exist_ok=True)
        
        # Save best model based on accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Store predictions for plotting
            best_test_true = test_true.copy()
            best_test_pred = test_pred.copy()
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_acc.t7'))
    
            with open(os.path.join(save_path, 'best_acc_epoch.txt'), 'w') as f:
                f.write(f'Best accuracy model saved at epoch {epoch}\n')
                f.write(f'Accuracy: {best_test_acc:.6f}')
            print(f"Best accuracy model saved with accuracy: {best_test_acc:.6f}")
    
    # Calculate and log total training time
    total_training_time = time.time() - total_training_start_time
    
    # Calculate additional metrics for best model
    best_balanced_acc = metrics.balanced_accuracy_score(best_test_true, best_test_pred)
    best_recall = metrics.recall_score(best_test_true, best_test_pred, average='macro')
    best_precision = metrics.precision_score(best_test_true, best_test_pred, average='macro')
    best_f1 = metrics.f1_score(best_test_true, best_test_pred, average='macro')
    print(f'Best test accuracy: {best_test_acc:.4f}')
    print(f'At best accuracy epoch - balanced acc: {best_balanced_acc:.4f}')
    print(f'At best accuracy epoch - precision: {best_precision:.4f}, recall: {best_recall:.4f}, f1: {best_f1:.4f}')
    print(f'Total training time: {total_training_time :.4f} s')
    
    # Add total training time to wandb summary
    wandb.run.summary["best_test_acc"] = best_test_acc*100
    # Add metrics at best accuracy epoch
    wandb.run.summary["balanced_acc_at_best_acc"] = best_balanced_acc*100
    wandb.run.summary["precision_at_best_acc"] = best_precision*100
    wandb.run.summary["recall_at_best_acc"] = best_recall*100
    wandb.run.summary["f1_at_best_acc"] = best_f1*100
    wandb.run.summary["total_training_time"] = total_training_time
    wandb.run.summary["time_per_epoch"] = total_training_time / args.epochs

    
    # Save final metrics summary
    with open(os.path.join(save_path, 'final_results.txt'), 'w') as f:
        f.write(f'Best test accuracy: {best_test_acc:.4f}\n')
        f.write(f'At best accuracy epoch - balanced acc: {best_balanced_acc:.4f}\n')
        f.write(f'At best accuracy epoch - precision: {best_precision:.4f}, recall: {best_recall:.4f}, f1: {best_f1:.4f}\n')
        f.write(f'Total training time: {total_training_time:.4f} s\n')
        
    # Save final model
    final_model_path = os.path.join(save_path, 'last.t7')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    results = model_interpreter(best_model, train_dataset.classes, save_dir='interpretation_analysis_after')
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

