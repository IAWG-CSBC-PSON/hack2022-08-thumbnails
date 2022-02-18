#!/usr/bin/env python
# coding: utf-8

import argparse
import tifffile
import zarr
import sys
import umap
import numpy as np
from colormath.color_objects import LabColor, sRGBColor, LCHuvColor, XYZColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import h5py
from pathlib import Path

import json
from scipy.cluster import hierarchy
from scipy.stats import norm
from functools import reduce
from uuid import uuid4

def auto_threshold(img):

    assert img.ndim == 2

    yi, xi = np.floor(np.linspace(0, img.shape, 200, endpoint=False)).astype(int).T
    # Slice one dimension at a time. Should generally use less memory than a meshgrid.
    img = img[yi]
    img = img[:, xi]
    img_log = np.log(img[img > 0])
    gmm = GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1,1)))
    means = gmm.means_[:, 0]
    _, i1, i2 = np.argsort(means)
    mean1, mean2 = means[[i1, i2]]
    std1, std2 = gmm.covariances_[[i1, i2], 0, 0] ** 0.5

    x = np.linspace(mean1, mean2, 50)
    y1 = norm(mean1, std1).pdf(x) * gmm.weights_[i1]
    y2 = norm(mean2, std2).pdf(x) * gmm.weights_[i2]

    lmax = mean2 + 2 * std2
    lmin = x[np.argmin(np.abs(y1 - y2))]
    if lmin >= mean2:
        lmin = mean2 - 2 * std2
    vmin = max(np.exp(lmin), img.min(), 0)
    vmax = min(np.exp(lmax), img.max())

    return vmin, vmax

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def pull_pyramid(input, level):
    
    print("Loading image")
    
    tiff = tifffile.TiffFile(input, is_ome=False)
    tiff_levels = tiff.series[0].levels
    highest_level_tiff = tiff_levels[level]
    zarray = zarr.open(highest_level_tiff.aszarr())

    print("Opened image pyramid level:", level)
    print("Image dimensions:", zarray.shape)

    return(zarray)
    
def remove_background(zarray):
    print("Finding background")
    sum_image = np.array(zarray).sum(axis = 0)
    pseudocount = 1
    log_image = np.log2(sum_image + pseudocount)
    thresh = threshold_otsu(log_image)
    binary = log_image > thresh
    cleaned = remove_small_objects(binary)
    print("Background removed")
    def get_tissue(x):
        return x[cleaned]
    tissue_array = list(map(get_tissue, zarray))
    tissue_array = np.array(tissue_array).T
    print("Selected", tissue_array.shape[0], "of", zarray.shape[1]*zarray.shape[2], "pixels as tissue")
    print("Pixels x channels matrix prepared")
    print(tissue_array.shape)
    return tissue_array,cleaned
    
def keep_background(zarray):
    print("Preserving background")
    shape = zarray.shape[1:]
    everything = np.ones(shape, dtype=bool)
    def get_all(x):
        return x[everything]
    tissue_array = list(map(get_all, zarray))
    tissue_array = np.array(tissue_array).T
    print("Pixels x channels matrix prepared")
    print(tissue_array.shape)
    return tissue_array,everything
    
def run_umap(tissue_array):
    reducer = umap.UMAP(
        n_components = 3,
        metric = "correlation",
        min_dist = 0,
        verbose = True)
    print("Running UMAP")
    embedding = reducer.fit_transform(tissue_array)
    return(embedding)
    
def run_tsne(tissue_array):
    reducer = TSNE(
        n_components = 3,
        metric = "correlation",
        square_distances = True,
        verbose = True)
    print("Running t-SNE")
    embedding = reducer.fit_transform(tissue_array)
    return(embedding)

def run_hclust(tissue_array, num_colors, color_method, dendrogram_file, color_index):

    #tissue_array = tissue_array[:10, :]

    [num_xy, num_channels] = tissue_array.shape

    # TODO: look into 2d convolution before clustering channels
    # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html

    # Perform hierarchical clustering along the channels axis.
    print("Hierarchically clustering")
    Z = hierarchy.linkage(tissue_array.T, method="ward")
    T = hierarchy.to_tree(Z)

    if dendrogram_file is not None:
        plt.figure()
        hierarchy.dendrogram(Z, link_color_func=lambda k: "k")
        plt.savefig(dendrogram_file)

    cutree = hierarchy.cut_tree(Z, n_clusters=num_colors).flatten()
    channel_mapping = zip(cutree, range(num_channels))
    channel_groups = dict()
    for k, v in channel_mapping:
        if k in channel_groups:
            channel_groups[k].append(v)
        else:
            channel_groups[k] = [v]

    print(channel_groups)

    scaler = MinMaxScaler(feature_range = (0.0,1.0))

    # Aggregate pixels for each channel group
    print("Aggregating pixels within groups")
    agg_array = np.zeros((num_xy, num_colors))
    for i, v in channel_groups.items():
        # TODO: explore alternative aggregation functions
        # TODO: maybe take weighted average based on information content of the channel (potentially after convolution)
        agg_array[:, i] = np.sum(tissue_array[:, v], axis=1)
        # normalize values for the aggregated channel
        # TODO: determine correct normalization/rescaling approach here
        agg_array[:, i] = (agg_array[:, i] - agg_array[:, i].min()) / agg_array[:, i].max()
    
    print(agg_array.shape)

    # Map values to to unique color per group
    print("Assigning colors")

    rgb_array = np.zeros((num_xy, 3))

    if color_method in {"pixel_sum", "pixel_max"}:
        
        for i in range(num_xy):
            if color_method == "pixel_sum":
                group_argmax = agg_array[i, :].argmax()
                if color_index is None or group_argmax == color_index:
                    lch = LCHuvColor(
                        lch_l = min(100, agg_array[i, :].sum()*100),
                        lch_c = 90,
                        lch_h = (group_argmax/num_colors)*360
                    )
                else:
                    lch = LCHuvColor(
                        lch_l = 0,
                        lch_c = 90,
                        lch_h = (group_argmax/num_colors)*360
                    )
            elif color_method == "pixel_max":
                lch = LCHuvColor(
                    lch_l = min(100, agg_array[i, :].max()*100),
                    lch_c = 90,
                    lch_h = (agg_array[i, :].argmax()/num_colors)*360
                )
            rgb = convert_color(lch, sRGBColor)
            clamped_rgb = sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
            clamped_rgb_arr = np.array(clamped_rgb.get_value_tuple())
            rgb_array[i, :] = clamped_rgb_arr
        
        rgb = rgb_array

    elif color_method == "group_hue":
        # Go to xyz space
        # Picking constant h for each cluster
        # XYZ is the central color space, then take average there, then convert to sRGB
        rgb_array = np.zeros((num_xy, 3))
        xyz_array = np.zeros((num_xy, num_colors, 3))

        for i in range(num_colors):
            lch = LCHuvColor(
                lch_l = 50,
                lch_c = 66,
                lch_h = (i/num_colors)*360
            )
            xyz = convert_color(lch, XYZColor)
            xyz_arr = np.array(xyz.get_value_tuple())

            xyz_array[:, i, :] = np.multiply(agg_array[:, i].repeat(3).reshape(-1, 3), np.tile(xyz_arr, num_xy).reshape(-1,3))

        # Do sum in XYZ space
        xyz = xyz_array.sum(axis=2)
        # Convert to sRGB
        for i in range(num_xy):
            xyz_obj = XYZColor(xyz_x = xyz[i, 0], xyz_y = xyz[i, 1], xyz_z = xyz[i, 2])
            rgb = convert_color(xyz_obj, sRGBColor)
            clamped_rgb = sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
            rgb_array[i, :] = np.array(clamped_rgb.get_value_tuple())
        
        rgb = rgb_array

    for i in range(3):
        # TODO: determine correct normalization/rescaling approach here
        # should normalization be done per-channel?
        rgb[:, i] = (rgb[:, i] - rgb[:, i].min()) / rgb[:, i].max()

    rgb = rgb.clip(min=0.0, max=1.0)

    return rgb
    
def embedding_to_lab_to_rgb(x):
        #print("Converting embedding to LAB colour")
        lab = LabColor(x[2], x[0], x[1])
        #print("Converting LAB to RGB for display")
        rgb = convert_color(lab, sRGBColor)
        #print("Clamping RGB values")
        clamped_rgb = sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)
        return clamped_rgb.get_value_tuple()
    
def assign_colours(embedding):
    print("Assigning colours to pixels embedding in low dimensional space")
    print("Rescaling embedding")
    scaler = MinMaxScaler(feature_range = (-128,127))
    dim1 = scaler.fit_transform(embedding[:,0].reshape(-1,1))
    dim2 = scaler.fit_transform(embedding[:,1].reshape(-1,1))
    scaler = MinMaxScaler(feature_range = (10,80))
    dim3 = scaler.fit_transform(embedding[:,2].reshape(-1,1))
    
    rescaled_embedding = np.concatenate((dim1,dim2,dim3), axis = 1)
    rescaled_embedding_list = rescaled_embedding.tolist()
    
    rgb = list(map(embedding_to_lab_to_rgb, rescaled_embedding_list))
    rgb = np.array(rgb)
    print("Colours assigned")
    return(rgb)
    
def make_rgb_image(rgb, mask):
    print("Painting miniature")
    rgb_shape = list(mask.shape)
    rgb_shape.append(3)
    rgb_image = np.zeros(rgb_shape)
    rgb_image[mask] = rgb
    return(rgb_image)

    
def plot_embedding(embedding, rgb, output):
    p = Path(output)
    newp = Path.joinpath(p.parent, p.stem+'-embedding' +p.suffix)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(
        embedding[:,0], 
        embedding[:,1], 
        embedding[:,2], 
        c = rgb,
        s = 20,
        edgecolors = 'none'
        )
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    plt.savefig(newp)
    
def save_data(output, args, tissue_array, mask, embedding, rgb, rgb_image):
    print("Saving log file")
    p = Path(output)
    p.rename(p.with_suffix('.h5'))
    h5 = h5py.File(p, 'w')
    #h5.create_dataset('args', data = args)
    h5.create_dataset('mask', data = mask)
    h5.create_dataset('tissue_array', data = tissue_array)
    h5.create_dataset('embedding', data = embedding)
    h5.create_dataset('rgb_array', data = rgb)
    h5.create_dataset('rgb_image', data = rgb_image)

def main():

    parser = argparse.ArgumentParser(description = 'Paint a miniature from an OME-TIFF')
    
    parser.add_argument('input',
                        type=str,
                        help=' a file name, seekable binary stream, or FileHandle for an OME-TIFF')
    
    parser.add_argument('output',
                        type=str,
                        default='data/miniature.png',
                        help='file name of output')
    
    parser.add_argument('-l', '--level',
                        type=int,
                        dest='level',
                        default=-1,
                        help='image pyramid level to use. defaults to -1 (highest)')
    
    parser.add_argument('-r', '--remove_bg',
                        type=str2bool,
                        dest='remove_bg',
                        default=True,
                        help='Attempt to remove background (defaults to True)')
                        
    parser.add_argument('--dimred',
                        type=str,
                        dest='dimred',
                        default='umap',
                        help='Dimensionality reduction method [umap, tsne]') 
    
    parser.add_argument('--num_colors',
                        type=int,
                        dest='num_colors',
                        default=3,
                        help='Number of colors for --dimred hclust')
    
    parser.add_argument('--color_method',
                        type=str,
                        dest='color_method',
                        default="pixel_sum",
                        help='Method of assigning colors to pixels for --dimred hclust')
    
    parser.add_argument('--color_index',
                        type=int,
                        dest='color_index',
                        default=None,
                        help='Only use one color channel for the output. This is a debugging option for --dimred hclust')
    
    parser.add_argument('--dendrogram_file',
                        type=str,
                        default=None,
                        dest='dendrogram_file',
                        help='File path for storing dendrogram plot for --dimred hclust')
                        
    parser.add_argument('--save_data',
                    type=str2bool,
                    dest='save_data',
                    default=False,
                    help='Save a h5 file with intermediate data')
                    
    parser.add_argument('--plot_embedding',
                    type=str2bool,
                    dest='plot_embedding',
                    default=False,
                    help='Save a figure of the embedding')
    
    args = parser.parse_args()
    
    zarray = pull_pyramid(args.input, args.level)
    
    if zarray.shape[0] == 3:
        rgb_image = np.moveaxis(zarray, 0, -1)
    else: 
        if args.remove_bg == False:
            tissue_array, mask = keep_background(zarray)
        elif args.remove_bg == True:
            tissue_array, mask = remove_background(zarray)
        else:
            tissue_array, mask = keep_background(zarray)
        
        if args.dimred == 'tsne':
            embedding = run_tsne(tissue_array)
            rgb = assign_colours(embedding)
        if args.dimred == 'umap':
            embedding = run_umap(tissue_array)
            rgb = assign_colours(embedding)
            print(rgb.shape)
        if args.dimred == 'hclust':
            rgb = run_hclust(tissue_array, args.num_colors, args.color_method, args.dendrogram_file, args.color_index-1 if args.color_index is not None else None)
        
        rgb_image = make_rgb_image(rgb, mask)
        
        print(rgb_image.shape)
    
    print("Saving image as " + args.output)
    output_path = args.output
    imsave(output_path, rgb_image)
    
    if args.save_data == True:
        save_data(output_path, args, tissue_array, mask, embedding, rgb, rgb_image)
    if args.plot_embedding == True:
        plot_embedding(embedding, rgb, output_path)

    print("Complete!")
    
if __name__ == "__main__":
    main()  
