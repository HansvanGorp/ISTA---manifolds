# Introduction
This is a codebase to experiment upon the geometry of (Learned) ISTA, which is the companion to the ICASSP submission under the title: "On the Geometry of (Learned) ISTA".

## Requirements
**Option 1: Docker**

You can build a docker image from the Dockerfile, or pull from dockerhub with:
```bash
docker pull oisinnolan/ista-geometry
```

**Option 2: Custom**
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.2+cu121 torchvision
pip install requirements.txt
```

## Scripts
For now, there is a main script that can be run and a simple dependent script for some post-hoc analysis. The main script that can be run is:
```
python main.py
```
This will start up an experiment as specified in "config_knot_density_experiment.yaml", which contains all the required settings. What it will do is set-up a number of trials for a certain experiment design (choice of N, M, K, etc.) and then train/evaluate ISTA, LISTA, and other variants for that experiment. The results are all saved in the subfolder "knot_denisty_results". This is the main way of using this repo, and all other files are in service to this main script. You can use the `--model_types` command line argument to specify which model types to train, e.g. `python main.py --model_types ISTA LISTA`.

After running a number of experiments using "main.py", one may want to plot a hyperplane through the input space of a partical model at a particular iterations. this can be done by running:
```
python plot_single_hyperplane.py
```
Which will do just that. The configuration file "config_plot_single_hyperplane.yaml" contains the settings for this which can be altered by the user.

## Reproducing Paper Results
#### **Hyperplane Visualizations**
- These can be created by running experiments using `main.py` and setting `Hyperplane['enabled']: True` in the `config.yaml`. Hyperplane plots will then be saved in the experiment results under `<experiment_dir>/<run_id>/<model_type>/hyperplane/jacobian_label`, e.g. `4_24_32_n=0.01_L2=0.0_1b83/0/LISTA/hyperplane/jacobian_label`.

#### **Controlling the knot density**

This experiment demonstrates that training LISTA with L2 regularization effectively determines the knot density.
Steps:
1. Run a sweep over L2 weights for both problem sizes:
```bash
python main.py --model_types LISTA --sweep_L2=True --config=/ISTA---manifolds/configs/main_experiments/4_24_32_n=0.01.yaml
python main.py --model_types LISTA --sweep_L2=True --config=/ISTA---manifolds/configs/main_experiments/8_64_64_n=0.01.yaml
```
2. When the experiments have finished, update the `BASE_EXPERIMENT_ROOTS` variable in `/ISTA---manifolds/plot_reg_weights_vs_knots.py` so that it points to the outputs of step 1.
3. Generate the plot using `python plot_reg_weights_vs_knots.py`

#### **Robustness to increased measurement noise**
1. Run a sweep over L2 weights for both problem sizes:
```bash
python main.py --model_types LISTA --sweep_L2=True --config=/ISTA---manifolds/configs/main_experiments/4_24_32_n=0.01.yaml
python main.py --model_types LISTA --sweep_L2=True --config=/ISTA---manifolds/configs/main_experiments/8_64_64_n=0.01.yaml
```
2. Run the following command to run the additional measurement noise experiments and generate plots:
```bash
python evaluate_ood_robustness.py --sweep_root=<path-to-experiment-output> --output_dir=<path-to-output-plots>
```
