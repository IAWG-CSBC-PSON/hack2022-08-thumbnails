# 08-thumbnails
Challenge 8: Unsupervised thumbnail generation for whole-slide multiplexed microscopy images

## Motivation
Image thumbnails provide users with rapid contextual information on imaging data in a small space. They also support the use of visual memory to recall individual interesting images from a large collection. Thumbnail generation strategies for brightfield (photographic) images are straightforward, but for highly multiplexed images with many channels and high dynamic range it is not immediately apparent how to optimally reduce the available information down to a small RGB image

## Goals
Participants will develop an approach to transform microscopy images in OME-TIFF format into thumbnail images stored as 300x300-pixel JPEG files. Input images will be as large as 50,000 pixels in the X and Y dimension and contain up to 40 discrete channels of 16-bit or 8-bit integer pixel data. Data from several different imaging technologies will be provided and data reduction approaches should work well with all of them. Participants may establish their own criteria and use cases for determining thumbnail image quality but must provide a rationale and justification for their choices. Solutions will be evaluated against the chosen quality criteria as well as runtime performance and resource usage.

## Example data
We have provided several example [OME-TIFF](https://docs.openmicroscopy.org/ome-model/6.2.0/ome-tiff/index.html) image files. The dimensions and file sizes vary greatly, but your solution should work for all of them. The image files are [tiled pyramids](https://docs.openmicroscopy.org/ome-model/6.2.2/ome-tiff/specification.html#sub-resolutions) to enable efficient data access -- the largest files are larger than available RAM on most personal computers.
https://www.synapse.org/#!Synapse:syn26858164

## Reference material

* **Miniature**: https://github.com/adamjtaylor/miniature/ -- Recolors high-dimensional images using UMAP to embed each pixel into CIELAB color space: . The repository is set up as a standard R project and the `docker/` subdirectory contains a Python port. You may wish to modify this code directly or simply use it as a reference.
