"""
This script will aggregate all individual results .csv files into a single table.
The script will search for all files in any subdirectory of 'results_dir' named 'results_file_name'.
The combined .csv file will be stored in 'output_table_path'.

e.g.
python generate_table_from_results.py \
    --results_dir=knot_denisty_results \
    --results_file_name=parameters.csv \
    --output_path=all_parameters.csv
"""

import os
import pandas as pd
import argparse

def find_csv_files(root_dir, target_filename):
    csv_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        if target_filename in filenames:
            csv_paths.append(os.path.join(dirpath, target_filename))
    return csv_paths

def combine_csv_files(csv_paths):
    combined_df = pd.DataFrame()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df['path'] = csv_path
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def save_combined_csv(combined_df, output_filename):
    combined_df.to_csv(output_filename, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Results table generation arguments")
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default="knot_denisty_results",
        help="Path to the results dir from which to generate the table.",
    )
    parser.add_argument(
        "-f",
        "--results_file_name",
        type=str,
        default="parameters.csv",
        help="Path to the results dir from which to generate the table.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="all_parameters.csv",
        help="Path to the results dir from which to generate the table.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    csv_paths = find_csv_files(args.results_dir, args.results_file_name)
    combined_df = combine_csv_files(csv_paths)
    save_combined_csv(combined_df, args.output_path)

    print(f"Combined CSV file saved as {args.output_path}")
