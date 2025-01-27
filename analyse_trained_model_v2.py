import os
import yaml
import torch
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

from knot_density_analysis import knot_density_analysis
from hyper_plane_analysis import visual_analysis_of_ista
from ista import ISTA, LISTA, ToeplitzLISTA
from analyse_trained_model import load_model


def analyze_trained_model(experiment_path, run_ID, model_name):
    """Analyze model's knot density and hyperplane structure"""
    
    run_path = experiment_path / run_ID

    # Load config and data
    with open(experiment_path / "config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    train_data = torch.load(run_path / "data/train_data.tar")
    
    # Load model
    model = load_model(
        config,
        model_name, 
        run_path / f"{model_name}/{model_name}_state_dict.tar",
        run_path / "A.tar",
        train_data,
        experiment_path
    )
    
    outdir = run_path / "analysis"
    outdir.mkdir(exist_ok=True)
    
    # Compute knot density
    knot_density = knot_density_analysis(
        model=model,
        nr_folds=config[model_name]["nr_folds"],
        A=model.A,
        nr_paths=config["Path"].get("nr_paths", 100),
        nr_points_along_path=config["Path"]["nr_points_along_path"],
        path_delta=config["Path"]["path_delta"],
        anchor_on_inputs=True,
        train_data=train_data,
        save_folder=outdir,
        save_name=f"knot_density_{model_name}",
        verbose=True
    )
    
    # Plot knot density
    plt.figure(figsize=(10, 6))
    plt.plot(knot_density, label=f"{model_name} Knot Density")
    plt.xlabel("Fold Number")
    plt.ylabel("Knot Density")
    plt.grid(True)
    plt.legend()
    plt.savefig(outdir / f"knot_density_{model_name}.pdf")
    plt.close()
    
    # Run hyperplane analysis
    if config.get("Hyperplane"):
        hyperplane_outdir = outdir / f"hyperplane_{model_name}"
        os.makedirs(hyperplane_outdir, exist_ok=True)
        visual_analysis_of_ista(
            model,
            config[model_name],
            config["Hyperplane"],
            model.A.cpu(),
            save_folder=hyperplane_outdir,
            verbose=True,
            color_by="jacobian_label",
            folds_to_visualize=range(config[model_name]["nr_folds"])
        )

if __name__ == "__main__":
    EXPERIMENT_ROOT = Path("/mnt/z/Ultrasound-BMd/data/oisin/knot_density_results/main_experiments/estimation_error/S/4_24_32_L2/4_24_32_n=0.01_L2=0.0_33e5")
    RUN_ID = "0"
    MODEL_NAME = "LISTA"
    analyze_trained_model(EXPERIMENT_ROOT, RUN_ID, MODEL_NAME)