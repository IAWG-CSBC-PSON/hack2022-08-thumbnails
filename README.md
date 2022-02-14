# 08-thumbnails
Challenge 8: Unsupervised thumbnail generation for whole-slide multiplexed microscopy images

## Motivation
Image thumbnails provide users with rapid contextual information on imaging data in a small space. They also support the use of visual memory to recall individual interesting images from a large collection. Thumbnail generation strategies for brightfield (photographic) images are straightforward, but for highly multiplexed images with many channels and high dynamic range it is not immediately apparent how to optimally reduce the available information down to a small RGB image

## Goals
Participants will develop an approach to transform microscopy images in OME-TIFF format into thumbnail images stored as 300x300-pixel JPEG files. Input images will be as large as 50,000 pixels in the X and Y dimension and contain up to 40 discrete channels of 16-bit integer or 32-bit floating point pixel data. Data from several different imaging technologies will be provided and data reduction approaches should work well with all of them. Participants may establish their own criteria and use cases for determining thumbnail image quality but must provide a rationale and justification for their choices. Solutions will be evaluated against the chosen quality criteria as well as runtime performance and resource usage.

## Example data
We have provided several example [OME-TIFF](https://docs.openmicroscopy.org/ome-model/6.2.0/ome-tiff/index.html) image files. The dimensions and file sizes vary greatly, but your solution should work for all of them. The image files are [tiled pyramids](https://docs.openmicroscopy.org/ome-model/6.2.2/ome-tiff/specification.html#sub-resolutions) to enable efficient data access -- the largest files are larger than available RAM on most personal computers.
https://www.synapse.org/#!Synapse:syn26858164

|Link|Name|X Size|Y Size|Channel Count|Pixel Data Type|Pixel Size (microns)|Channel Names|
|----|----|-|-|-|-|-|-|
|[syn26947045](https://www.synapse.org/#!Synapse:syn26947045)|cycif_colorectal_carcinoma.ome.tif|26139|27120|40|uint16|0.65|DNA,Autofluorescence-488nm,Autofluorescence-555nm,Autofluorescence-647nm,DNA (2),Control-488nm,Control-555nm,Control-647nm,DNA (3),CD3,Na/K ATPase,CD45RO,DNA (4),Antigen Ki67,Pan-cytokeratin,Aortic smooth muscle actin,DNA (5),CD4,CD45,PD-1,DNA (6),CD20,CD68,CD8a,DNA (7),CD163,FOXP3,PD-L1,DNA (8),E-cadherin,Vimentin,CDX-2,DNA (9),Lamin-A/B/C,Desmin,CD31,DNA (10),PCNA,Antigen Ki67 (2),Collagen|
|[syn26947033](https://www.synapse.org/#!Synapse:syn26947033)|cycif_tma.ome.tif|6197|6231|40|uint16|0.65|DNA_1,AF488,AF555,AF647,DNA_2,A488_background,A555_background,A647_background,DNA_3,FDX1,CD357,CD1D,DNA_4,CD163,CD3D,CD31,DNA_5,LDH,CD66B,VDAC1,DNA_6,ELANE,CD57,CD45,DNA_7,CD11B,SMA,CD16,DNA_8,ECAD,FOXP3,NCAM,DNA_9,CD4,KERATIN,CD14,DNA_10,IBA1,CD1B,CD8A|
|[syn26946496](https://www.synapse.org/#!Synapse:syn26946496)|cycif_tonsil.ome.tif|3500|2500|9|uint16|0.325|DNA,Ki-67,Keratin,CD3D,CD4,CD45,CD8A,α-SMA,CD20|
|[syn26858183](https://www.synapse.org/#!Synapse:syn26858183)|mibi_liver.ome.tiff|1024|1024|27|float32||beta-tubulin, CD11b, CD11c, CD163, CD20, CD3, CD31, CD4, CD45, CD45RO, CD56, CD68, CD8, DC-SIGN, dsDNA, FOXP3, Granzyme_B, HLA_class_1_A_B_and_C_Na-K-ATPase_alpha1, HLA_DR, IDO-1, Keratin, Ki-67, LAG3, PD-1, PD-L1, Podoplanin, Vimentin|
|[syn26858168](https://www.synapse.org/#!Synapse:syn26858168)|mibi_placenta.ome.tiff|1024|1024|27|float32||beta-tubulin, CD11b, CD11c, CD163, CD20, CD3, CD31, CD4, CD45, CD45RO, CD56, CD68, CD8, DC-SIGN, dsDNA, FOXP3, Granzyme_B, HLA_class_1_A_B_and_C_Na-K-ATPase_alpha1, HLA_DR, IDO-1, Keratin, Ki-67, LAG3, PD-1, PD-L1, Podoplanin, Vimentin|
|[syn26858167](https://www.synapse.org/#!Synapse:syn26858167)|mibi_thymus.ome.tiff|2048|2048|27|float32||beta-tubulin, CD11b, CD11c, CD163, CD20, CD3, CD31, CD4, CD45, CD45RO, CD56, CD68, CD8, DC-SIGN, dsDNA, FOXP3, Granzyme_B, HLA_class_1_A_B_and_C_Na-K-ATPase_alpha1, HLA_DR, IDO-1, Keratin, Ki-67, LAG3, PD-1, PD-L1, Podoplanin, Vimentin|
|[syn26858166](https://www.synapse.org/#!Synapse:syn26858166)|mibi_tonsil.ome.tiff|2048|2048|27|float32|1.25|beta-tubulin, CD11b, CD11c, CD163, CD20, CD3, CD31, CD4, CD45, CD45RO, CD56, CD68, CD8, DC-SIGN, dsDNA, FOXP3, Granzyme_B, HLA_class_1_A_B_and_C_Na-K-ATPase_alpha1, HLA_DR, IDO-1, Keratin, Ki-67, LAG3, PD-1, PD-L1, Podoplanin, Vimentin|
|[syn26858194](https://www.synapse.org/#!Synapse:syn26858194)|mibi_tumor_FOV1.ome.tiff|1024|1024|24|float32||beta-tubulin, CD11b, CD11c, CD163, CD20, CD3, CD31, CD4, CD45, CD56, CD68, CD8, dsDNA, FOXP3, HLA_class_1_A_B_and_C_Na-K-ATPase_alpha1, HLA_DR, IDO-1, Keratin, Ki-67, LAG3, PD-1, PD-L1, Podoplanin, Vimentin|
|[syn26858193](https://www.synapse.org/#!Synapse:syn26858193)|mibi_tumor_FOV3.ome.tiff|1024|1024|24|float32||beta-tubulin, CD11b, CD11c, CD163, CD20, CD3, CD31, CD4, CD45, CD56, CD68, CD8, dsDNA, FOXP3, HLA_class_1_A_B_and_C_Na-K-ATPase_alpha1, HLA_DR, IDO-1, Keratin, Ki-67, LAG3, PD-1, PD-L1, Podoplanin, Vimentin|
|[syn26858192](https://www.synapse.org/#!Synapse:syn26858192)|mibi_tumor_FOV5.ome.tiff|1024|1024|24|float32||beta-tubulin, CD11b, CD11c, CD163, CD20, CD3, CD31, CD4, CD45, CD56, CD68, CD8, dsDNA, FOXP3, HLA_class_1_A_B_and_C_Na-K-ATPase_alpha1, HLA_DR, IDO-1, Keratin, Ki-67, LAG3, PD-1, PD-L1, Podoplanin, Vimentin|

|Image|Channel Montage|Miniature|
|---|---|---|
|CyCIF Colorectal Carcinoma|![cycif_colorectal_carcinoma](https://user-images.githubusercontent.com/14945787/153918768-a7aec271-eaa5-4414-959a-194aab049c1e.png)|![thumbnail](https://user-images.githubusercontent.com/14945787/153918819-45d48f00-f0d8-477c-a759-e78ef0abcecb.png)|
|CyCIF Tonsil| ![tonsil](https://user-images.githubusercontent.com/14945787/153916476-eba282b3-3f34-4277-8467-50349c764840.png) |![thumbnail](https://user-images.githubusercontent.com/14945787/153916549-f910ffe5-9ef4-464d-8106-e7713286cfca.png)|
|MIBI Tumor FOV1|![mibi_tumor1_montage](https://user-images.githubusercontent.com/14945787/153917922-80f07ca2-390e-4919-b762-3d47a86f4392.png)|![thumbnail](https://user-images.githubusercontent.com/14945787/153918240-296918b3-371f-4b9a-97cf-719c36196329.png)|




## Reference material

* **Miniature** (https://github.com/adamjtaylor/miniature/): Recolors high-dimensional images using UMAP to embed each pixel into CIELAB color space. The repository is set up as a standard R project and the `docker/` subdirectory contains a Python port. You may wish to modify this code directly or simply use it as a reference. ![image](https://user-images.githubusercontent.com/14945787/127400268-b6345cf4-a90c-4d77-9f83-6889de6763a5.png)

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
