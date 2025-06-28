import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
def set_font_size(size=10):
    plt.rcParams.update({
        'font.size': size,
        'font.family': 'serif',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def load_and_process_data(csv_file):
    """Load and process the ablation study data"""
    df = pd.read_csv(csv_file)
    
    # Display basic info
    print("DataFrame shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nUnique parameters:", sorted(df['Parameter_Varied'].unique()))
    print("\nData preview:")
    print(df.head())
    
    return df

def get_parameter_label(param_type, layer_num):
    """Get properly formatted parameter label with majuscule P_i+1"""
    # Convert layer numbers (2,3,4,5) to display numbers (1,2,3,4)
    display_num = layer_num - 1
    return f'{param_type.upper()}_{display_num}'

def extract_layer_and_param(param_name):
    """Extract layer number and parameter type from parameter name"""
    # Handle parameters with underscore (like g_2, w_3, s_4, sm_5, wtp_2)
    if '_' in param_name:
        parts = param_name.split('_')
        param_type = parts[0]
        layer_num = int(parts[1])
    # Handle parameters without underscore (like g2, w3, s4, sm5, wtp2)
    elif param_name.startswith('wtp'):
        param_type = 'wtp'
        layer_num = int(param_name[3:])
    elif param_name.startswith('sm'):
        param_type = 'sm'
        layer_num = int(param_name[2:])
    else:
        param_type = param_name[0]  # g, w, s
        layer_num = int(param_name[1:])
    
    # Map layer numbers to layer names - indices are incremented by 1
    # So parameter index 2=ECL1, 3=ECL2, 4=FEL, 5=CL
    layer_mapping = {2: 'ECL1', 3: 'ECL2', 4: 'FEL', 5: 'CL'}
    layer_name = layer_mapping.get(layer_num, f'Layer{layer_num}')
    
    return layer_name, param_type, layer_num

def create_parameter_performance_plots(df, font_size=10):
    """Create compact plots grouped by parameter type for publication"""
    set_font_size(font_size)
    
    # Get unique parameters and group by type
    df_processed = df.copy()
    layer_param_info = df_processed['Parameter_Varied'].apply(extract_layer_and_param)
    df_processed['Layer'] = [info[0] for info in layer_param_info]
    df_processed['ParamType'] = [info[1] for info in layer_param_info]
    
    # Only process numeric parameters (g, w, s) - exclude sm and wtp
    numeric_param_types = ['g', 'w', 's']
    
    # Create separate plots for each numeric parameter type
    for param_type in numeric_param_types:
        if param_type in df_processed['ParamType'].unique():
            create_numeric_parameter_summary(df_processed, param_type, font_size)

def create_numeric_parameter_summary(param_data, param_type, font_size):
    """Create publication-ready plot for numeric parameters"""
    # Filter data for this parameter type
    param_data = param_data[param_data['ParamType'] == param_type].copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
    layers = ['ECL1', 'ECL2', 'FEL', 'CL']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, layer in enumerate(layers):
        layer_data = param_data[param_data['Layer'] == layer]
        if len(layer_data) > 0:
            layer_data = layer_data.sort_values('Parameter_Value')
            # Get layer number for proper labeling
            layer_info = layer_data.iloc[0]
            _, _, layer_num = extract_layer_and_param(layer_info['Parameter_Varied'])
            param_label = get_parameter_label(param_type, layer_num)
            
            ax.plot(layer_data['Parameter_Value'], layer_data['Accuracy'], 
                   marker='o', linewidth=1.5, markersize=4, color=colors[i], 
                   label=f'{param_label}', alpha=0.8)
    
    ax.set_xlabel(f'Parameter Value', fontsize=font_size)
    ax.set_ylabel('Accuracy', fontsize=font_size)
    ax.legend(loc='best', fontsize=font_size-2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(f'Parameter {param_type.upper()} Sensitivity', 
                fontsize=font_size+1, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'parameter_{param_type}_analysis.pdf', format='pdf')
    plt.show()

def create_best_performance_heatmap(df, font_size=10):
    """Create compact heatmap for publication"""
    set_font_size(font_size)
    
    # Calculate best performance for each parameter
    best_performance = df.groupby('Parameter_Varied')['Accuracy'].max().reset_index()
    
    # Extract layer and parameter information
    results = []
    for _, row in best_performance.iterrows():
        layer_name, param_type, layer_num = extract_layer_and_param(row['Parameter_Varied'])
        
        # Include all parameter types (g, w, s, sm, wtp) for heatmap
        if param_type in ['g', 'w', 's', 'sm', 'wtp']:
            # Map parameter types to display names
            param_display = {
                'g': 'G',
                'w': 'W', 
                's': 'S',
                'sm': 'SM',
                'wtp': 'WT'
            }
            results.append({
                'Layer': layer_name,
                'ParamType': param_display[param_type],
                'Layer_Num': layer_num,
                'Best_Accuracy': row['Accuracy'],
                'Original_Param': row['Parameter_Varied']
            })
    
    results_df = pd.DataFrame(results)
    
    # Check if we have data
    if len(results_df) == 0:
        print("Warning: No parameters found for heatmap")
        return results_df
    
    # Create pivot table for heatmap using ParamType
    try:
        pivot_df = results_df.pivot(index='Layer', columns='ParamType', values='Best_Accuracy')
        
        # Reorder layers correctly
        layer_order = ['ECL1', 'ECL2', 'FEL', 'CL']
        available_layers = [layer for layer in layer_order if layer in pivot_df.index]
        pivot_df = pivot_df.reindex(available_layers)
        
        # Reorder parameters in logical order
        param_order = ['G', 'W', 'S', 'SM', 'WT']
        available_params = [param for param in param_order if param in pivot_df.columns]
        pivot_df = pivot_df[available_params]
        
        # Handle missing values - fill with NaN for better visualization
        # Don't fill with 0 as it's misleading for missing data
        
        # Create compact heatmap
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        
        # Check if pivot_df has valid data
        if pivot_df.empty or pivot_df.isna().all().all():
            ax.text(0.5, 0.5, 'No data available for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Best Performance by Layer and Parameter', 
                        fontsize=font_size+1, fontweight='bold')
        else:
            # Use only valid values for center calculation
            valid_values = pivot_df.values[~pd.isna(pivot_df.values)]
            center_val = valid_values.mean() if len(valid_values) > 0 else 0.5
            
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                        center=center_val, 
                        square=True, 
                        cbar_kws={'label': 'Best Accuracy', 'shrink': 0.8},
                        linewidths=0.5, ax=ax,
                        cbar=True)
            
            ax.set_title('Best Performance by Layer and Parameter', 
                        fontsize=font_size+1, fontweight='bold')
            ax.set_xlabel('Parameter Type', fontsize=font_size)
            ax.set_ylabel('Layer', fontsize=font_size)
        
        plt.tight_layout()
        plt.savefig('performance_heatmap.pdf', format='pdf')
        plt.show()
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        print("Available data:")
        print(results_df)
        
        # Create a simple fallback plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Heatmap Error', fontsize=font_size+1, fontweight='bold')
        plt.tight_layout()
        plt.savefig('performance_heatmap.pdf', format='pdf')
        plt.show()
    
    return results_df

def create_parameter_importance_ranking(df, font_size=10):
    """Create compact parameter importance plot for publication"""
    set_font_size(font_size)
    
    # Calculate parameter statistics
    param_stats = df.groupby('Parameter_Varied')['Accuracy'].agg(['min', 'max', 'mean', 'std']).reset_index()
    param_stats['range'] = param_stats['max'] - param_stats['min']
    
    # Add proper parameter labels
    param_labels = []
    for param in param_stats['Parameter_Varied']:
        layer_name, param_type, layer_num = extract_layer_and_param(param)
        param_label = get_parameter_label(param_type, layer_num)
        param_labels.append(param_label)
    param_stats['Param_Label'] = param_labels
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Top parameters by performance
    top_performers = param_stats.nlargest(8, 'max').sort_values('max', ascending=True)
    bars1 = ax1.barh(range(len(top_performers)), top_performers['max'], alpha=0.7, color='#1f77b4')
    ax1.set_yticks(range(len(top_performers)))
    ax1.set_yticklabels(top_performers['Param_Label'], fontsize=font_size-1)
    ax1.set_xlabel('Best Accuracy', fontsize=font_size)
    ax1.set_title('Top Parameters by Performance', fontsize=font_size+1, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linewidth=0.5)
    
    # Add value labels
    for i, v in enumerate(top_performers['max']):
        ax1.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=font_size-2)
    
    # Plot 2: Most sensitive parameters
    most_sensitive = param_stats.nlargest(8, 'range').sort_values('range', ascending=True)
    bars2 = ax2.barh(range(len(most_sensitive)), most_sensitive['range'], 
                     alpha=0.7, color='#ff7f0e')
    ax2.set_yticks(range(len(most_sensitive)))
    ax2.set_yticklabels(most_sensitive['Param_Label'], fontsize=font_size-1)
    ax2.set_xlabel('Performance Range', fontsize=font_size)
    ax2.set_title('Most Sensitive Parameters', fontsize=font_size+1, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linewidth=0.5)
    
    # Add value labels
    for i, v in enumerate(most_sensitive['range']):
        ax2.text(v + 0.0005, i, f'{v:.3f}', va='center', fontsize=font_size-2)
    
    plt.tight_layout()
    plt.savefig('parameter_importance.pdf', format='pdf')
    plt.show()
    
    return param_stats

def create_layer_comparison_boxplot(df, font_size=10):
    """Create compact layer comparison for publication"""
    set_font_size(font_size)
    
    # Extract layer information
    df_processed = df.copy()
    layer_param_info = df_processed['Parameter_Varied'].apply(extract_layer_and_param)
    df_processed['Layer'] = [info[0] for info in layer_param_info]
    df_processed['ParamType'] = [info[1] for info in layer_param_info]
    
    # Filter only numeric parameters
    df_processed = df_processed[df_processed['ParamType'].isin(['g', 'w', 's'])]
    
    # Create compact box plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    
    # Custom color palette for only g, w, s
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    sns.boxplot(data=df_processed, x='Layer', y='Accuracy', hue='ParamType', 
                palette=colors, ax=ax)
    ax.set_title('Parameter Performance by Layer', fontsize=font_size+1, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=font_size)
    ax.set_ylabel('Accuracy', fontsize=font_size)
    ax.legend(title='Parameter', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-1)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('layer_comparison.pdf', format='pdf')
    plt.show()

def create_parameter_impact_analysis(df, font_size=10):
    """Create plots showing how G, W, S parameter values impact model size and training time"""
    set_font_size(font_size)
    
    # Check if required columns exist
    required_cols = ['Num_Parameters', 'Training_Time_Min', 'Parameter_Value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for parameter impact analysis: {missing_cols}")
        return
    
    # Process data
    df_processed = df.copy()
    layer_param_info = df_processed['Parameter_Varied'].apply(extract_layer_and_param)
    df_processed['Layer'] = [info[0] for info in layer_param_info]
    df_processed['ParamType'] = [info[1] for info in layer_param_info]
    
    # Convert parameter count to millions
    df_processed['Num_Parameters_M'] = df_processed['Num_Parameters'] / 1_000_000
    
    # Filter only numeric parameters (g, w, s)
    numeric_data = df_processed[df_processed['ParamType'].isin(['g', 'w', 's'])].copy()
    
    if len(numeric_data) == 0:
        print("Warning: No numeric parameters (g, w, s) found for parameter impact analysis")
        return
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Color mapping for parameter types
    colors = {'g': '#1f77b4', 'w': '#ff7f0e', 's': '#2ca02c'}
    param_names = {'g': 'G', 'w': 'W', 's': 'S'}
    
    # Top row: Parameter count vs Parameter value
    # Bottom row: Training time vs Parameter value
    
    for i, param_type in enumerate(['g', 'w', 's']):
        param_data = numeric_data[numeric_data['ParamType'] == param_type]
        
        if len(param_data) == 0:
            # Hide empty subplots
            axes[0, i].set_visible(False)
            axes[1, i].set_visible(False)
            continue
        
        # Sort by parameter value for better line plots
        param_data = param_data.sort_values('Parameter_Value')
        
        # Top row: Parameter Count vs Parameter Value
        ax_top = axes[0, i]
        
        # Group by layer for different lines
        for layer in ['ECL1', 'ECL2', 'FEL', 'CL']:
            layer_data = param_data[param_data['Layer'] == layer]
            if len(layer_data) > 0:
                ax_top.plot(layer_data['Parameter_Value'], layer_data['Num_Parameters_M'],
                           marker='o', linewidth=2, markersize=5, label=layer, alpha=0.8)
        
        ax_top.set_title(f'Parameter {param_names[param_type]}: Model Size Impact', 
                        fontsize=font_size, fontweight='bold')
        ax_top.set_xlabel(f'{param_names[param_type]} Value', fontsize=font_size-1)
        ax_top.set_ylabel('Parameters (Millions)', fontsize=font_size-1)
        ax_top.grid(True, alpha=0.3, linewidth=0.5)
        ax_top.legend(fontsize=font_size-2)
        
        # Bottom row: Training Time vs Parameter Value
        ax_bottom = axes[1, i]
        
        # Group by layer for different lines
        for layer in ['ECL1', 'ECL2', 'FEL', 'CL']:
            layer_data = param_data[param_data['Layer'] == layer]
            if len(layer_data) > 0:
                ax_bottom.plot(layer_data['Parameter_Value'], layer_data['Training_Time_Min'],
                              marker='s', linewidth=2, markersize=5, label=layer, alpha=0.8)
        
        ax_bottom.set_title(f'Parameter {param_names[param_type]}: Training Time Impact', 
                           fontsize=font_size, fontweight='bold')
        ax_bottom.set_xlabel(f'{param_names[param_type]} Value', fontsize=font_size-1)
        ax_bottom.set_ylabel('Training Time (Minutes)', fontsize=font_size-1)
        ax_bottom.grid(True, alpha=0.3, linewidth=0.5)
        ax_bottom.legend(fontsize=font_size-2)
    
    plt.tight_layout()
    plt.savefig('parameter_impact_analysis.pdf', format='pdf')
    plt.show()
    
    # Print analysis summary
    print("\n" + "="*60)
    print("PARAMETER IMPACT ANALYSIS SUMMARY")
    print("="*60)
    
    for param_type in ['g', 'w', 's']:
        param_data = numeric_data[numeric_data['ParamType'] == param_type]
        if len(param_data) == 0:
            continue
            
        print(f"\n{param_names[param_type]} Parameter Analysis:")
        print(f"  Value range: {param_data['Parameter_Value'].min()} - {param_data['Parameter_Value'].max()}")
        
        # Calculate correlations
        size_corr = param_data['Parameter_Value'].corr(param_data['Num_Parameters_M'])
        time_corr = param_data['Parameter_Value'].corr(param_data['Training_Time_Min'])
        
        print(f"  Model size correlation: {size_corr:.3f}")
        print(f"  Training time correlation: {time_corr:.3f}")
        
        # Find optimal values
        best_efficiency = param_data.loc[param_data['Accuracy'].idxmax()]
        layer_name, _, layer_num = extract_layer_and_param(best_efficiency['Parameter_Varied'])
        param_label = get_parameter_label(param_type, layer_num)
        
        print(f"  Best performing: {param_label} = {best_efficiency['Parameter_Value']} "
              f"(Acc: {best_efficiency['Accuracy']:.4f}, "
              f"Size: {best_efficiency['Num_Parameters_M']:.2f}M, "
              f"Time: {best_efficiency['Training_Time_Min']:.2f}min)")
    
    return numeric_data

def create_parameter_efficiency_plot(df, font_size=10):
    """Create scatter plot showing parameter count vs training time for different parameters"""
    set_font_size(font_size)
    
    # Check if required columns exist
    required_cols = ['Num_Parameters', 'Training_Time_Min']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for efficiency plot: {missing_cols}")
        return
    
    # Process data
    df_processed = df.copy()
    layer_param_info = df_processed['Parameter_Varied'].apply(extract_layer_and_param)
    df_processed['Layer'] = [info[0] for info in layer_param_info]
    df_processed['ParamType'] = [info[1] for info in layer_param_info]
    
    # Convert parameter count to millions
    df_processed['Num_Parameters_M'] = df_processed['Num_Parameters'] / 1_000_000
    
    # Map parameter types to display names
    param_display = {
        'g': 'G', 'w': 'W', 's': 'S', 'sm': 'SM', 'wtp': 'WT'
    }
    df_processed['ParamDisplay'] = df_processed['ParamType'].map(param_display)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Color palette for different parameter types
    colors = {'G': '#1f77b4', 'W': '#ff7f0e', 'S': '#2ca02c', 'SM': '#d62728', 'WT': '#9467bd'}
    
    # Plot points for each parameter type
    for param_type in ['G', 'W', 'S', 'SM', 'WT']:
        if param_type in df_processed['ParamDisplay'].values:
            param_data = df_processed[df_processed['ParamDisplay'] == param_type]
            ax.scatter(param_data['Num_Parameters_M'], param_data['Training_Time_Min'],
                      c=colors[param_type], label=param_type, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    if len(df_processed) > 1:
        z = np.polyfit(df_processed['Num_Parameters_M'], df_processed['Training_Time_Min'], 1)
        p = np.poly1d(z)
        ax.plot(df_processed['Num_Parameters_M'], p(df_processed['Num_Parameters_M']), 
                "r--", alpha=0.5, linewidth=1.5, label='Trend')
    
    # Formatting
    ax.set_xlabel('Number of Parameters (Millions)', fontsize=font_size)
    ax.set_ylabel('Training Time (Minutes)', fontsize=font_size)
    ax.set_title('Parameter Efficiency: Model Size vs Training Time', 
                fontsize=font_size+1, fontweight='bold')
    ax.legend(fontsize=font_size-1, title='Parameter Type')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add some statistics as text
    correlation = df_processed['Num_Parameters_M'].corr(df_processed['Training_Time_Min'])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=font_size-1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('parameter_efficiency.pdf', format='pdf')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("PARAMETER EFFICIENCY ANALYSIS")
    print("="*50)
    print(f"Parameter count range: {df_processed['Num_Parameters_M'].min():.2f}M - {df_processed['Num_Parameters_M'].max():.2f}M")
    print(f"Training time range: {df_processed['Training_Time_Min'].min():.2f} - {df_processed['Training_Time_Min'].max():.2f} minutes")
    print(f"Correlation (parameters vs time): {correlation:.3f}")
    
    # Find most efficient configurations
    df_processed['Efficiency'] = df_processed['Accuracy'] / (df_processed['Num_Parameters_M'] * df_processed['Training_Time_Min'] / 1000)
    top_efficient = df_processed.nlargest(3, 'Efficiency')[['Parameter_Varied', 'Accuracy', 'Num_Parameters_M', 'Training_Time_Min', 'Efficiency']]
    
    print(f"\nTOP 3 MOST EFFICIENT CONFIGURATIONS:")
    for i, (_, row) in enumerate(top_efficient.iterrows()):
        layer_name, param_type, layer_num = extract_layer_and_param(row['Parameter_Varied'])
        param_label = get_parameter_label(param_type, layer_num)
        print(f"{i+1}. {param_label}: Acc={row['Accuracy']:.4f}, Params={row['Num_Parameters_M']:.2f}M, Time={row['Training_Time_Min']:.2f}min")
    
    return df_processed
    """Create compact layer comparison for publication"""
    set_font_size(font_size)
    
    # Extract layer information
    df_processed = df.copy()
    layer_param_info = df_processed['Parameter_Varied'].apply(extract_layer_and_param)
    df_processed['Layer'] = [info[0] for info in layer_param_info]
    df_processed['ParamType'] = [info[1] for info in layer_param_info]
    
    # Filter only numeric parameters
    df_processed = df_processed[df_processed['ParamType'].isin(['g', 'w', 's'])]
    
    # Create compact box plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    
    # Custom color palette for only g, w, s
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    sns.boxplot(data=df_processed, x='Layer', y='Accuracy', hue='ParamType', 
                palette=colors, ax=ax)
    ax.set_title('Parameter Performance by Layer', fontsize=font_size+1, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=font_size)
    ax.set_ylabel('Accuracy', fontsize=font_size)
    ax.legend(title='Parameter', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-1)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('layer_comparison.pdf', format='pdf')
    plt.show()

def generate_summary_report(df, param_stats):
    """Generate a summary report of the ablation study"""
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY REPORT")
    print("="*60)
    
    print(f"\nTotal experiments conducted: {len(df)}")
    print(f"Parameters tested: {len(df['Parameter_Varied'].unique())}")
    print(f"Overall accuracy range: {df['Accuracy'].min():.4f} - {df['Accuracy'].max():.4f}")
    
    print("\nTOP 5 BEST PERFORMING PARAMETERS:")
    top_5 = param_stats.nlargest(5, 'max')[['Param_Label', 'max']].reset_index(drop=True)
    for i, row in top_5.iterrows():
        print(f"{i+1}. {row['Param_Label']}: {row['max']:.4f}")
    
    print("\nMOST SENSITIVE PARAMETERS (highest performance range):")
    top_sensitive = param_stats.nlargest(5, 'range')[['Param_Label', 'range']].reset_index(drop=True)
    for i, row in top_sensitive.iterrows():
        print(f"{i+1}. {row['Param_Label']}: {row['range']:.4f}")
    
    print("\nMOST STABLE PARAMETERS (lowest performance range):")
    most_stable = param_stats.nsmallest(5, 'range')[['Param_Label', 'range']].reset_index(drop=True)
    for i, row in most_stable.iterrows():
        print(f"{i+1}. {row['Param_Label']}: {row['range']:.4f}")

def main(font_size=10):
    """Main function to run all analyses and generate publication-ready figures"""
    # Load data
    print("Loading ablation study data...")
    df = load_and_process_data('ablation.csv')
    
    # Create all visualizations
    print("\n1. Creating parameter performance plots (numeric parameters only)...")
    create_parameter_performance_plots(df, font_size)
    
    print("\n2. Creating best performance heatmap...")
    results_df = create_best_performance_heatmap(df, font_size)
    
    print("\n3. Creating parameter importance ranking...")
    param_stats = create_parameter_importance_ranking(df, font_size)
    
    print("\n4. Creating layer comparison boxplots...")
    create_layer_comparison_boxplot(df, font_size)
    
    print("\n5. Creating parameter efficiency plot...")
    efficiency_df = create_parameter_efficiency_plot(df, font_size)
    
    print("\n6. Creating parameter impact analysis...")
    impact_df = create_parameter_impact_analysis(df, font_size)
    
    print("\n7. Generating summary report...")
    generate_summary_report(df, param_stats)
    
    print("\n" + "="*60)
    print("PUBLICATION-READY FIGURES GENERATED!")
    print("Files saved:")
    print("- parameter_g_analysis.pdf")
    print("- parameter_w_analysis.pdf") 
    print("- parameter_s_analysis.pdf")
    print("- performance_heatmap.pdf")
    print("- parameter_importance.pdf")
    print("- layer_comparison.pdf")
    print("- parameter_efficiency.pdf")
    print("- parameter_impact_analysis.pdf")
    print("="*60)

if __name__ == "__main__":
    # You can control font size here (default is 10)
    main(font_size=12)  # Change to your desired font size
