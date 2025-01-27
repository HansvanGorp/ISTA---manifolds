import os
import re
import yaml
from pathlib import Path

# Function to extract the regularization weight from the directory name
def extract_weight_from_name(dir_name):
    # Regular expression to match "L2_w=X", where X is the weight (a float or integer)
    match = re.search(r'L2_w=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', dir_name)
    if match:
        return float(match.group(1))
    return None

# Function to update the YAML file with the regularization weight
def update_yaml_with_weight(yaml_path, weight):
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update the specific YAML key with the weight
        if 'LISTA' in config:
            config['LISTA']['regularization'] = {'weight': weight}
        
        # Write the updated configuration back to the file
        with open(yaml_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"Updated {yaml_path} with weight: {weight}")
    
    except Exception as e:
        print(f"Failed to update {yaml_path}: {e}")

# Main function to traverse directories and update YAML files
def update_weights_in_directory(directory_path):
    # Convert to Path object for easy traversal
    directory = Path(directory_path)
    
    if not directory.is_dir():
        print(f"{directory_path} is not a valid directory.")
        return
    
    # Traverse the directory for subdirectories
    for subdir in directory.iterdir():
        if subdir.is_dir():  # Only process directories
            weight = extract_weight_from_name(subdir.name)
            if weight is not None:
                # Construct the path to the config.yaml file in the subdirectory
                yaml_file = subdir / 'config.yaml'
                if yaml_file.exists():
                    update_yaml_with_weight(yaml_file, weight)
                else:
                    print(f"No config.yaml found in {subdir}")

# Example usage
if __name__ == "__main__":
    directory_path = '/ISTA---manifolds/knot_denisty_results/sweep_noise_levels/8_64_64/8_64_64_n=0.01_0d6d/post_training_L2'
    update_weights_in_directory(directory_path)
