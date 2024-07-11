#!/bin/bash
sweep_name=sweep_cloud_scale
configs_dir=/app/configs/sweep_cloud_scale
results_dir=/app/knot_denisty_results/test_L2_loss_M2_N3_nrtrain_8192_sweep_cloud_scale
id=$(date '+%Y-%m-%d_%H:%M:%S')

echo "Running sweep ${sweep_name} with id ${id}"

for config_path in "$configs_dir"/*.yaml; do 
    python main.py --config=$config_path --model_types=RLISTA
done

python generate_table_from_results.py \
    --results_dir=$results_dir \
    --results_file_name=parameters.csv \
    --output_path="all_parameters_${sweep_name}_${id}.csv"

mkdir "output/${sweep_name}_${id}"

python generate_parallel_plots_from_results_table.py \
    --results_table_path="all_parameters_${sweep_name}_${id}.csv" \
    --output_dir="output/${sweep_name}_${id}"