#!/usr/bin/env python
# coding: utf-8

import tifffile
import bokeh.palettes as bp
import zarr
import numpy as np

import os
import argparse

import imageio as iio
from PIL import ImageColor

# pull lower resolution image from OME-TIFF file
def pull_pyramid(input, level):
    tiff = tifffile.TiffFile(input, is_ome=False)
    tiff_levels = tiff.series[0].levels
    highest_level_tiff = tiff_levels[level]
    zarray = zarr.open(highest_level_tiff.aszarr())

    return zarray

# parse CLI input for folder location
def get_cli_input():
    parser = argparse.ArgumentParser(description='Make a preivew GIF from an OME-TIFF')

    parser.add_argument('input',
                        type=str,
                        help=' a single OME-TIFF file or a folder with OME-TIFF files')

    return parser.parse_args()

# add bokeh color palette
def grayscale_to_rgb(channel, color):
    r = channel * color[0]
    g = channel * color[1]
    b = channel * color[2]

    return (np.dstack((r, g, b))).astype(np.uint8)


# gif assembly function
def convert_tiff_to_gif(file_name):

    # get the zarray
    zarray = pull_pyramid(file_name, -2)

    # generate rgb colors
    hex_colors = bp.magma(zarray.shape[0])
    rgb_colors = [ImageColor.getcolor(hex, "RGB") for hex in hex_colors]

    gif_frames = []
    for i in range(len(zarray)):
        # apply rgb
        gif_frames.append(grayscale_to_rgb(zarray[i], rgb_colors[ len(rgb_colors) % (i+1)]))

    # assemble into gif
    file_name = file_name.split('\\')
    file_name = file_name[len(file_name) - 1]
    file_name = file_name.split('.')[0]
    path = 'OME-TIFF Previews/' + file_name + '.gif'
    iio.mimsave(path, gif_frames, fps=1)

def main():

    args = get_cli_input()
    files = os.listdir(args.input)

    # setup preview folder if it doesnt exist
    try:
        os.mkdir('OME-TIFF Previews')
        print("Made output folder")
    except:
        pass

    print("Making all previews from folder")
    print("This may take a minute...")
    for file in files:
        path = os.path.join(args.input, file)
        convert_tiff_to_gif(path)

    print("Done!")


if __name__ == "__main__":
    main()
