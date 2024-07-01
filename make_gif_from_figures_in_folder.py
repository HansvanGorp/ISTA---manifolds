"""
This script creates a gif from all png images in a folder. frames are ordered by alphabetical name of the figure files.
"""

import imageio
import os
from natsort import natsorted

def make_gif_from_figures_in_folder(folder, animation_total_seconds):
    # get all files in folder
    files = os.listdir(folder)

    # sort files
    files = natsorted(files)

    # get all files with .png extension
    files = [f for f in files if f.endswith('.png')]

    # read all images
    images = []
    for filename in files:
        images.append(imageio.imread(os.path.join(folder, filename)))


    save_path = os.path.join(folder, 'animation.gif')
    imageio.mimsave(save_path, images, duration=animation_total_seconds/len(images))