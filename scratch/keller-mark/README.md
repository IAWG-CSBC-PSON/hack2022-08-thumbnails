# A scratch space for @keller-mark

## Setup

Create and activate the conda environment.

```sh
conda env create -f environment.yml
conda activate thumbnails-env
```

To enable the Snakefile to download raw data files, set ] Synapse credentials in the following environment variables:

```sh
export SYNAPSE_USERNAME="my_username_here"
export SYNAPSE_PASSWORD="mY-sUpEr-SeCrEt-pAsSwOrD-HeRe"
```

## Downloading data

```sh
snakemake --cores 1
```

## Testing miniature

```sh
python ../../miniature/paint_miniature.py \
    'data/raw/download/cycif_tonsil.ome.tif' \
    'data/processed/miniature.jpg'
```


```sh
python ../../miniature/paint_miniature.py \
    'data/raw/download/cycif_tonsil.ome.tif' \
    'data/processed/miniature.jpg' \
    --remove_bg False --dimred hclust
```