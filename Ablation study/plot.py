import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np

def generate_model_plots(test_true, test_pred, num_classes, model_name, accuracy, exp_id, class_names=None):
    """
    Generate high-quality ROC curves and confusion matrix for a model.
    """
    
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
