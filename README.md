# Introduction
This is a codebase to experiment upon the geometry of (Learned) ISTA, which is the companion to the ICASSP submission under the title: "On the Geometry of (Learned) ISTA".

## Requirements
The requirements for now are quite bloated, as they contain many packages that are not necesarrily needed to run everyhting. For now, I specify a simple environment to exactly reproduce my results on windows-64. I am yet to make this exportable to other platforms, as that can be quite finnicky.

To create a conda environment on windows, you can run:
```
conda create --name ista_env --file spec-file.txt
```
Then activate the environment as:
```
conda activate ista_env
```

## Scripts that can be run
For now, there is a main script that can be run and a simple dependent script for some post-hoc analysis. The main script that can be run is:
```
python main.py
```
This will start up an experiment as specified in "config_knot_density_experiment.yaml", which contains all the required settings. What it will do is set-up a number of trials for a certain experiment design (choice of N, M, K, etc.) and then train/evaluate ISTA, LISTA, and RLISTA for that experiment. The results are all saved in the subfolder "knot_denisty_results". This is the main way of using this repo, and all other files are in service to this main script.

After running a number of experiments using "main.py", one may want to plot a hyperplane through the input space of a partical model at a particular iterations. this can be done by running:
```
python plot_single_hyperplane.py
```
Which will do just that. The configuration file "config_plot_single_hyperplane.yaml" contains the settings for this which can be altered by the user.

## short explenation of each file
I here list each file with a very brief decription of what they contain:
- "data_on_plane.py": This script defines a class to find all occurences of data on a hyperplane embedded in a higher dimensional space.
- "data.py": This file creates the functions that create the data and the dataloaders for the experiments
- "experiment_design.py": This file creates some functions usefull for the design of experiments.
- "hyper_plane_analysis.py": This script creates the functions used to analyze the linear regions of (RL)ISTA along a hyperplane.
- "ista.py": We here implement (R)(L)ISTA as a pytorch module
- "knot_density_analysis.py": This script creates the functions used to analyze the knot density of the ISTA algorithm.
- "main.py": main script, see explanation above.
- "make_gig_from_figures_in_folder.py": This script creates a gif from all png images in a folder. frames are ordered by alphabetical name of the figure files.
- "ncolor.py": Implementation of ncolor theorm in python from: https://forum.image.sc/t/relabel-with-4-colors-like-map/33564/6
- "parallel_coordinates.py": make a parralel coordinate plot from a dataframe
- "plot_single_hyperplane.py": This script will load a single experiment as specified in the config file and plot the hyperplane of the model.
- "README.md": the readme file you are currently reading.
- "spec-file.txt": the anaconda specification file to recreate the environment I use on a windows machine
- "training.py": This file specifies different functions to train (R)(L)ISTA modules