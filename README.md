# 08-thumbnails
Challenge 8: Unsupervised thumbnail generation for whole-slide multiplexed microscopy images

## Motivation
Image thumbnails provide users with rapid contextual information on imaging data in a small space. They also support the use of visual memory to recall individual interesting images from a large collection. Thumbnail generation strategies for brightfield (photographic) images are straightforward, but for highly multiplexed images with many channels and high dynamic range it is not immediately apparent how to optimally reduce the available information down to a small RGB image

## Goals
Participants will develop an approach to transform microscopy images in OME-TIFF format into thumbnail images stored as 300x300-pixel JPEG files. Input images will be as large as 50,000 pixels in the X and Y dimension and contain up to 40 discrete channels of 16-bit or 8-bit integer pixel data. Data from several different imaging technologies will be provided and data reduction approaches should work well with all of them. Participants may establish their own criteria and use cases for determining thumbnail image quality but must provide a rationale and justification for their choices. Solutions will be evaluated against the chosen quality criteria as well as runtime performance and resource usage.

## Example data
We have provided several example [OME-TIFF](https://docs.openmicroscopy.org/ome-model/6.2.0/ome-tiff/index.html) image files. The dimensions and file sizes vary greatly, but your solution should work for all of them. The image files are [tiled pyramids](https://docs.openmicroscopy.org/ome-model/6.2.2/ome-tiff/specification.html#sub-resolutions) to enable efficient data access -- the largest files are larger than available RAM on most personal computers.
https://www.synapse.org/#!Synapse:syn26858164

|Link|Name|
|----|----|
|syn26947045|cycif_colorectal_carcinoma.ome.tif|
|syn26947033|cycif_tma.ome.tif|
|syn26946496|cycif_tonsil.ome.tif|
|syn26858183|mibi_liver.ome.tiff|
|syn26858168|mibi_placenta.ome.tiff|
|syn26858167|mibi_thymus.ome.tiff|
|syn26858166|mibi_tonsil.ome.tiff|
|syn26858194|mibi_tumor_FOV1.ome.tiff|
|syn26858193|mibi_tumor_FOV3.ome.tiff|
|syn26858192|mibi_tumor_FOV5.ome.tiff|


## Reference material

* **Miniature** (https://github.com/adamjtaylor/miniature/): Recolors high-dimensional images using UMAP to embed each pixel into CIELAB color space: . The repository is set up as a standard R project and the `docker/` subdirectory contains a Python port. You may wish to modify this code directly or simply use it as a reference. ![image](https://user-images.githubusercontent.com/14945787/127400268-b6345cf4-a90c-4d77-9f83-6889de6763a5.png)

## Tools

Useful python packages include `tifffile`, `imagecodecs`, `scikit-image`, `umap-learn`, `zarr` and `colormath`. You may wish to setup a Conda environemt with recomended modules,

```
wget https://raw.githubusercontent.com/adamjtaylor/htan-artist/main/docker/environment.yml
conda env create -n artist --file=environment.yaml
```

or use the `adamjtaylor/htan-artist` docker container with these installed. Eg:
```
docker run -it --rm --platform linux/amd64 -v $HOME/Documents/projects/csbc/hack2022-08-thumbnails/data:/data adamjtaylor/htan-artist
```

## Other resources

You will want a viewer capable of loading and displaying the example images. We recommend either [Napari](https://napari.org/) or [ImageJ / Fiji](https://fiji.sc/).
