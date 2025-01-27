# read df

# split into two (multiple?) dfs by model_type

# for each model_type df:

# group by noise_std, get mean and stderr columns

# plot with matplotlib the noise_std on x-axis, train_loss_end and test_loss_end on y-axis

# save the plot to png and pdf

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

colors = {
    "ISTA": "tab:blue", 
    "FISTA": "tab:red",
    "LISTA": "tab:orange", 
    "RLISTA": "tab:green",
    "ToeplitzLISTA": "tab:olive"
}

# Specify the path to your CSV file
csv_file_path = Path("/ISTA---manifolds/result_analysis/results/all_params_noise_sweep_8_64_64.csv")
output_dir = Path("/ISTA---manifolds/result_analysis/plots")

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

model_types = df['model_type'].unique()

# Plot 'train_loss_end' and 'test_loss_end' against 'noise_std'
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Two side-by-side plots

# Loop through each model_type DataFrame
for model_type in model_types:
    # Filter the DataFrame by the current model_type
    model_df = df[df['model_type'] == model_type]

    # Group by 'noise_std' and calculate mean and stderr for 'train_loss_end' and 'test_loss_end'
    grouped_df = model_df.groupby('noise_std').agg(
        train_loss_end_mean=('train_loss_end', 'mean'),
        train_loss_end_stderr=('train_loss_end', 'sem'),
        test_loss_end_mean=('test_loss_end', 'mean'),
        test_loss_end_stderr=('test_loss_end', 'sem')
    ).reset_index()

    # Calculate 95% confidence intervals
    grouped_df['train_loss_end_stderr_95ci'] = 1.96 * grouped_df['train_loss_end_stderr']
    grouped_df['test_loss_end_stderr_95ci'] = 1.96 * grouped_df['test_loss_end_stderr']
    
    # Left plot: Train and test loss
    ax1 = axes[0]
    ax1.errorbar(grouped_df['noise_std'], grouped_df['train_loss_end_mean'], 
                 yerr=grouped_df['train_loss_end_stderr_95ci'], label=f'{model_type} Train', color=colors[model_type], linestyle="dashed")
    ax1.errorbar(grouped_df['noise_std'], grouped_df['test_loss_end_mean'], 
                 yerr=grouped_df['test_loss_end_stderr_95ci'], label=f'{model_type} Test', color=colors[model_type], linestyle="solid")
    
    # Plot filled areas for 95% confidence intervals
    ax1.fill_between(grouped_df['noise_std'], 
                    grouped_df['train_loss_end_mean'] - grouped_df['train_loss_end_stderr_95ci'], 
                    grouped_df['train_loss_end_mean'] + grouped_df['train_loss_end_stderr_95ci'], 
                    color=colors[model_type], alpha=0.2)
    ax1.fill_between(grouped_df['noise_std'], 
                    grouped_df['test_loss_end_mean'] - grouped_df['test_loss_end_stderr_95ci'], 
                    grouped_df['test_loss_end_mean'] + grouped_df['test_loss_end_stderr_95ci'], 
                    color=colors[model_type], alpha=0.2)
    ax1.set_title('Train and Test Loss')
    ax1.set_xlabel('Noise Std')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Right plot: Knot density
    grouped_knot_density_df = model_df.groupby('noise_std').agg(
        knot_density_end_mean=('knot_density_end', 'mean'),
        knot_density_end_stderr=('knot_density_end', 'sem')
    ).reset_index()
    
    grouped_knot_density_df['knot_density_end_stderr_95ci'] = 1.96 * grouped_knot_density_df['knot_density_end_stderr']
    
    ax2 = axes[1]
    ax2.errorbar(grouped_knot_density_df['noise_std'], grouped_knot_density_df['knot_density_end_mean'], 
                 yerr=grouped_knot_density_df['knot_density_end_stderr_95ci'], label=f'{model_type} Knot Density', color=colors[model_type], linestyle="solid")
    
    # Plot filled areas for knot density 95% confidence intervals
    ax2.fill_between(grouped_knot_density_df['noise_std'], 
                     grouped_knot_density_df['knot_density_end_mean'] - grouped_knot_density_df['knot_density_end_stderr_95ci'], 
                     grouped_knot_density_df['knot_density_end_mean'] + grouped_knot_density_df['knot_density_end_stderr_95ci'], 
                     color=colors[model_type], alpha=0.2)
    
    ax2.set_title('Knot Density')
    ax2.set_xlabel('Noise Std')
    ax2.set_ylabel('Knot Density')
    ax2.legend()


ax1.set_title('Train and Test Loss')
ax1.set_xlabel('Noise Std')
ax1.set_ylabel('MAE')
ax1.grid(True)
ax1.legend()

ax2.set_title('Knot Density')
ax2.set_xlabel('Noise Std')
ax2.set_ylabel('Knot Density')
ax2.grid(True)
ax2.legend()

# Save the plot as both PNG and PDF
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
plt.tight_layout()
csv_name = os.path.splitext(os.path.basename(csv_file_path))[0]
plt.savefig(output_dir / f'{csv_name}.png')
plt.savefig(output_dir / f'{csv_name}.pdf')
print('OUTPUT TO: ', output_dir / f'{csv_name}.png')

# Close the figure to avoid overlapping plots in the next iteration
plt.close()
