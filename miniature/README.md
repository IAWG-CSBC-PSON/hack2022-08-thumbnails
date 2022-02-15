Core `paint_miniature.py` script taken from [`adamjtaylor/miniature`](https://github.com/adamjtaylor/miniature) commit `1d411d1`

```
python paint_miniature.py 'data/HTA9_1_BA_L_ROI04.ome.tif' 'miniature.jpg'

```

Optional arguments allow for changing level used, preserving background, saving the 3D embedding plot, and saving the intermediate data (tissue mask, data matrix, embedding and colours as h5. Optionally t-SNE can be used but this is slower than UMAP

For example, to paint a miniature on the second higest level, preserving the background, using t-SNE and saving both the 3D embedding and intermediate data use

```
python paint_miniature.py 'data/HTA9_1_BA_L_ROI04.ome.tif' 'miniature.jpg' \
     --level -2 --remove_bg True, --dimred tsne --save_data True --plot_embedding True
````
