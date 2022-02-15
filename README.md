# 08-thumbnails
Challenge 8: Unsupervised thumbnail generation for whole-slide multiplexed microscopy images

## Motivation
Image thumbnails provide users with rapid contextual information on imaging data in a small space. They also support the use of visual memory to recall individual interesting images from a large collection. Thumbnail generation strategies for brightfield (photographic) images are straightforward, but for highly multiplexed images with many channels and high dynamic range it is not immediately apparent how to optimally reduce the available information down to a small RGB image

## Goals
Participants will develop an approach to transform microscopy images in OME-TIFF format into thumbnail images stored as 300x300-pixel JPEG files. Input images will be as large as 50,000 pixels in the X and Y dimension and contain up to 40 discrete channels of 16-bit or 8-bit integer pixel data. Data from several different imaging technologies will be provided and data reduction approaches should work well with all of them. Participants may establish their own criteria and use cases for determining thumbnail image quality but must provide a rationale and justification for their choices. Solutions will be evaluated against the chosen quality criteria as well as runtime performance and resource usage.

## Example data
We have provided several example [OME-TIFF](https://docs.openmicroscopy.org/ome-model/6.2.0/ome-tiff/index.html) image files. The dimensions and file sizes vary greatly, but your solution should work for all of them. The image files are [tiled pyramids](https://docs.openmicroscopy.org/ome-model/6.2.2/ome-tiff/specification.html#sub-resolutions) to enable efficient data access -- the largest files are larger than available RAM on most personal computers.
https://www.synapse.org/#!Synapse:syn26858164


### Data download
You can download data into a local data directory as follows

Install the [`synapseclient`](http://python-docs.synapse.org/build/html/index.html#installation) Python package and cli tool
>The synapseclient package is available from PyPI. It can be installed or upgraded with pip. Note that synapseclient requires Python 3, and if you have both Python 2 and Python 3 installations your system, the pip command associated with Python 3 may be named pip3 to distinguish it from a Python 2 associated command. Prefixing the pip installation with sudo may be necessary if you are installing Python into a shared system installation of Python. The dependencies on pandas and pysftp are optional. The Synapse synapseclient.table feature integrates with Pandas. Support for sftp is required for users of SFTP file storage. Both require native libraries to be compiled or installed separately from prebuilt binaries.

```
(sudo) pip3 install (--upgrade) synapseclient[pandas, pysftp]
```

Create a Synapse config file in `~/.synapseConfig` following the instructions (TBC)

Download all example data
```
mkdir data
cd data
synapse get syn26858164 -r
```


## Reference material

* **Miniature** (https://github.com/adamjtaylor/miniature/): Recolors high-dimensional images using UMAP to embed each pixel into CIELAB color space: . The repository is set up as a standard R project and the `docker/` subdirectory contains a Python port. You may wish to modify this code directly or simply use it as a reference. ![image](https://user-images.githubusercontent.com/14945787/127400268-b6345cf4-a90c-4d77-9f83-6889de6763a5.png)

## Other resources

You will want a viewer capable of loading and displaying the example images. We recommend either [Napari](https://napari.org/) or [ImageJ / Fiji](https://fiji.sc/).
