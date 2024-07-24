"""
This script will make a set of parallel plots from a csv file of the
'parameters.csv' format. The parallel plots will be generated per
model_type.

e.g.
python generate_parallel_plots_from_results_table.py \
    --results_table_path=all_parameters.csv \
    --output_dir=output
"""

import os
import uuid
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from parallel_coordinates import plot_df_as_parallel_coordinates

def parse_args():
    parser = argparse.ArgumentParser(description="Results table generation arguments")
    parser.add_argument(
        "-r",
        "--results_table_path",
        type=str,
        default="all_parameters.csv",
        help="Path to the csv file containing the results table.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="output",
        help="Directory in which to save the plots.",
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    # short ID generator to prevent overwriting
    id = lambda: str(uuid.uuid4())[:4]

    df = pd.read_csv(args.results_table_path)

    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type]
        fig, ax = plt.subplots(1,1, figsize=(16,8))
        plot_df_as_parallel_coordinates(model_df, 
                                        ["knot_density_max",  "knot_density_end", "test_loss_end", "test_accuracy_end"], 
                                        "test_accuracy_end",
                                        host = ax, title=model_type, accuracy_scale_collumns = ["test_accuracy_end"],
                                        same_y_scale_collumns = [["knot_density_max",  "knot_density_end"]])    
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"parallel_coordinates_{model_type}_{id()}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(args.output_dir, f"parallel_coordinates_{model_type}_{id()}.svg"), bbox_inches='tight')
        plt.close()