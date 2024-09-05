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