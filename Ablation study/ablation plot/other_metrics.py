import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.rcParams.update({
    'font.size': 10,
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
    layer_name = layer_mapping.get(layer_num, f'Layer_{layer_num}')
    
    return layer_name, param_type, layer_num

def analyze_best_parameters(csv_file, top_n=10):
    """Comprehensive analysis of best performing parameters with all metrics"""
    
    print("="*80)
    print("🎯 BEST PARAMETERS DETAILED ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} experiments from {csv_file}")
    
    # Check available columns
    required_base_cols = ['Parameter_Varied', 'Parameter_Value', 'Accuracy']
    optional_cols = ['Recall', 'F1_Score', 'Num_Parameters', 'Training_Time_Min']
    
    available_cols = [col for col in optional_cols if col in df.columns]
    missing_cols = [col for col in optional_cols if col not in df.columns]
    
    print(f"\nAvailable metrics: {['Accuracy'] + available_cols}")
    if missing_cols:
        print(f"Missing metrics: {missing_cols}")
    
    # Hardcode epochs to 300 as requested
    EPOCHS = 300
    print(f"Using hardcoded epochs: {EPOCHS}")
    
    # Process parameter information
    df_processed = df.copy()
    layer_param_info = df_processed['Parameter_Varied'].apply(extract_layer_and_param)
    df_processed['Layer'] = [info[0] for info in layer_param_info]
    df_processed['ParamType'] = [info[1] for info in layer_param_info]
    df_processed['Layer_Num'] = [info[2] for info in layer_param_info]
    
    # Add formatted parameter labels
    df_processed['Param_Label'] = df_processed.apply(
        lambda row: get_parameter_label(row['ParamType'], row['Layer_Num']), axis=1
    )
    
    # Get top N performing configurations
    top_configs = df_processed.nlargest(top_n, 'Accuracy')
    
    print(f"\n🏆 TOP {top_n} BEST PERFORMING PARAMETER CONFIGURATIONS:")
    print("="*80)
    
    # Create detailed results table
    results = []
    for i, (_, row) in enumerate(top_configs.iterrows()):
        result = {
            'Rank': i + 1,
            'Parameter': row['Param_Label'],
            'Value': str(row['Parameter_Value']),  # Convert to string to handle any data type
            'Layer': row['Layer'],
            'Accuracy': float(row['Accuracy'])  # Ensure it's a float
        }
        
        # Add available metrics with proper formatting and type handling
        for col in available_cols:
            if col == 'Num_Parameters':
                # Convert to millions for readability
                try:
                    if pd.notna(row[col]):
                        params_val = float(row[col])
                        result['Params_M'] = f"{params_val / 1_000_000:.2f}M"
                    else:
                        result['Params_M'] = 'N/A'
                except (ValueError, TypeError):
                    result['Params_M'] = 'N/A'
                    
            elif col == 'Training_Time_Min':
                # Calculate epoch time: Training_Time_Min * 60 / EPOCHS (in seconds)
                try:
                    if pd.notna(row[col]):
                        time_val = float(row[col])
                        epoch_time_sec = (time_val * 60) / EPOCHS
                        result['Time_Min'] = f"{time_val:.2f} min"
                        result['Epoch_Time'] = f"{epoch_time_sec:.1f} sec/epoch"
                    else:
                        result['Time_Min'] = 'N/A'
                        result['Epoch_Time'] = 'N/A'
                except (ValueError, TypeError):
                    result['Time_Min'] = 'N/A'
                    result['Epoch_Time'] = 'N/A'
                    
            else:
                # Handle Recall, F1_Score, etc.
                try:
                    if pd.notna(row[col]):
                        metric_val = float(row[col])
                        result[col] = f"{metric_val:.4f}"
                    else:
                        result[col] = 'N/A'
                except (ValueError, TypeError):
                    result[col] = 'N/A'
        
        results.append(result)
    
    # Display results
    for result in results:
        print(f"\n{result['Rank']:2d}. {result['Parameter']} = {result['Value']} (Layer: {result['Layer']})")
        
        # Create formatted output
        metrics_line = f"    📊 Accuracy: {result['Accuracy']:.4f}"
        
        if 'Recall' in result and result['Recall'] != 'N/A':
            metrics_line += f" | Recall: {result['Recall']}"
        if 'F1_Score' in result and result['F1_Score'] != 'N/A':
            metrics_line += f" | F1: {result['F1_Score']}"
        
        print(metrics_line)
        
        # Resource information
        if 'Params_M' in result and result['Params_M'] != 'N/A':
            resource_line = f"    🔧 Parameters: {result['Params_M']}"
            if 'Time_Min' in result and result['Time_Min'] != 'N/A':
                resource_line += f" | Training: {result['Time_Min']}"
            if 'Epoch_Time' in result and result['Epoch_Time'] != 'N/A':
                resource_line += f" | Per Epoch: {result['Epoch_Time']}"
            print(resource_line)
    
    return create_best_params_visualization(results, available_cols)

def create_best_params_visualization(results, available_cols):
    """Create visualization of best parameters analysis"""
    
    if len(results) == 0:
        print("No results to visualize")
        return
    
    # Prepare data for plotting
    plot_data = []
    for result in results:
        plot_data.append({
            'Rank': result['Rank'],
            'Parameter': result['Parameter'],
            'Accuracy': result['Accuracy'],
            'Value': result['Value']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy ranking
    ax1 = axes[0]
    bars = ax1.barh(range(len(plot_df)), plot_df['Accuracy'], alpha=0.7, color='#1f77b4')
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels([f"{row['Parameter']} = {row['Value']}" for _, row in plot_df.iterrows()], 
                        fontsize=9)
    ax1.set_xlabel('Accuracy', fontsize=10)
    ax1.set_title('Top Parameter Configurations by Accuracy', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, plot_df['Accuracy'])):
        ax1.text(bar.get_width() + 0.001, i, f'{acc:.4f}', 
                va='center', fontsize=8)
    
    # Invert y-axis to show rank 1 at top
    ax1.invert_yaxis()
    
    # Plot 2: Parameter value distribution
    ax2 = axes[1]
    param_types = [p.split('_')[0] for p in plot_df['Parameter']]
    unique_types = list(set(param_types))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    color_map = {ptype: colors[i] for i, ptype in enumerate(unique_types)}
    
    bar_colors = [color_map[ptype] for ptype in param_types]
    bars2 = ax2.bar(range(len(plot_df)), plot_df['Value'], alpha=0.7, color=bar_colors)
    ax2.set_xticks(range(len(plot_df)))
    ax2.set_xticklabels([f"#{row['Rank']}" for _, row in plot_df.iterrows()], fontsize=9)
    ax2.set_ylabel('Parameter Value', fontsize=10)
    ax2.set_xlabel('Rank', fontsize=10)
    ax2.set_title('Parameter Values of Top Configurations', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, plot_df['Value'])):
        ax2.text(i, bar.get_height() + max(plot_df['Value']) * 0.01, f'{val}', 
                ha='center', va='bottom', fontsize=8)
    
    # Create legend for parameter types
    legend_elements = [plt.Rectangle((0,0),1,1, color=color_map[ptype], alpha=0.7, label=ptype) 
                      for ptype in unique_types]
    ax2.legend(handles=legend_elements, title='Parameter Type', loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('best_parameters_analysis.pdf', format='pdf')
    plt.show()
    
    return plot_df

def create_efficiency_comparison(csv_file, top_n=10):
    """Compare efficiency metrics of top parameters"""
    
    df = pd.read_csv(csv_file)
    
    # Check for required columns
    required_cols = ['Accuracy', 'Num_Parameters', 'Training_Time_Min']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing required columns for efficiency comparison")
        return
    
    # Process data
    df_processed = df.copy()
    layer_param_info = df_processed['Parameter_Varied'].apply(extract_layer_and_param)
    df_processed['Layer'] = [info[0] for info in layer_param_info]
    df_processed['ParamType'] = [info[1] for info in layer_param_info]
    df_processed['Layer_Num'] = [info[2] for info in layer_param_info]
    
    df_processed['Param_Label'] = df_processed.apply(
        lambda row: get_parameter_label(row['ParamType'], row['Layer_Num']), axis=1
    )
    
    # Calculate efficiency metrics with proper type handling
    try:
        df_processed['Params_M'] = pd.to_numeric(df_processed['Num_Parameters'], errors='coerce') / 1_000_000
        df_processed['Training_Time'] = pd.to_numeric(df_processed['Training_Time_Min'], errors='coerce')
        df_processed['Accuracy_Val'] = pd.to_numeric(df_processed['Accuracy'], errors='coerce')
        
        # Calculate efficiency score: Higher is better
        df_processed['Efficiency_Score'] = (df_processed['Accuracy_Val'] * 1000) / (df_processed['Params_M'] * df_processed['Training_Time'])
        
        # Remove rows with NaN values
        df_processed = df_processed.dropna(subset=['Efficiency_Score', 'Accuracy_Val'])
        
    except Exception as e:
        print(f"Error calculating efficiency metrics: {e}")
        return
    
    # Get top performers by different metrics
    top_accuracy = df_processed.nlargest(top_n, 'Accuracy_Val')
    top_efficiency = df_processed.nlargest(top_n, 'Efficiency_Score')
    
    print(f"\n📈 EFFICIENCY COMPARISON ANALYSIS")
    print("="*60)
    
    print(f"\n🏆 TOP {min(5, top_n)} BY ACCURACY:")
    for i, (_, row) in enumerate(top_accuracy.head(5).iterrows()):
        print(f"{i+1}. {row['Param_Label']} = {row['Parameter_Value']}: "
              f"Acc={row['Accuracy_Val']:.4f}, Params={row['Params_M']:.2f}M, "
              f"Time={row['Training_Time']:.2f}min")
    
    print(f"\n⚡ TOP {min(5, top_n)} BY EFFICIENCY:")
    for i, (_, row) in enumerate(top_efficiency.head(5).iterrows()):
        print(f"{i+1}. {row['Param_Label']} = {row['Parameter_Value']}: "
              f"Acc={row['Accuracy_Val']:.4f}, Params={row['Params_M']:.2f}M, "
              f"Time={row['Training_Time']:.2f}min, Score={row['Efficiency_Score']:.2f}")
    
    return df_processed

def main():
    """Main function to run best parameters analysis"""
    csv_file = 'ablation.csv'
    
    try:
        # Run detailed analysis
        print("Running best parameters analysis...")
        results_df = analyze_best_parameters(csv_file, top_n=10)
        
        print("\n" + "="*80)
        
        # Run efficiency comparison
        print("Running efficiency comparison...")
        efficiency_df = create_efficiency_comparison(csv_file, top_n=10)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("📄 Generated file: best_parameters_analysis.pdf")
        print("="*80)
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
