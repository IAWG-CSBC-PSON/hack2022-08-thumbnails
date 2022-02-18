import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

### k-means on umap
def kmeans_rgb(embedding, rgb):
    clusternum=3
    # kmeans = KMeans(n_clusters=clusternum, random_state=0).fit(tissue_array)
    kmeans = KMeans(n_clusters=clusternum, random_state=0).fit(embedding)
    label = kmeans.labels_
    # kmeans.cluster_centers_

    for j in range(len(label)):
        if label[j] == 0:
            rgb[j, :] = [0,0,1]
        if label[j] == 1:
            rgb[j, :] = [0,1,0]
        if label[j] == 2:
            rgb[j, :] = [1,0,0]
    return rgb, label

## differential analysis
def diff(tissue_array,label,cluster):
    ratio_cluster=[]
    for j in range(len(tissue_array[0,])):
        l = []
        l_other=[]
        for i in range(len(tissue_array)):
            if label[i]==cluster:
                l.append(tissue_array[i,j])
            elif label[i]!=cluster:
                l_other.append(tissue_array[i,j])
        mean_l=sum(l)/len(l)
        mean_l_other=sum(l_other)/len(l_other)
        rati=mean_l/mean_l_other
        ratio_cluster.append(rati)
    return ratio_cluster

def marker_label(tissue_array):
    label=[]
    for i in range(len(tissue_array[0])):
        label1 = np.percentile(tissue_array[:,i], 70)
        binary=tissue_array[:,i]>label1
        label.append(binary)
    label=np.array(label)
    return label

def rgb_marker_all(rgb,marker_binary,marker_num):
    for j in range(len(rgb)):
        if marker_binary[marker_num][j]==0:
            rgb[j]=np.ones((1,3))
    return rgb

def rgb_marker_img(rgb,marker_binary,marker_num):
    for j in range(len(marker_binary[marker_num])):
        if marker_binary[marker_num][j]==0:
            rgb[j,2]=0.5
            rgb[j, 1] = 0.5
            rgb[j, 0] = 0.5
    return rgb


############ logic pipeline ############

input='cycif_tonsil_small.tiff'

level=0
zarray = pull_pyramid(input, level)
tissue_array, mask = remove_background(zarray)
marker_binary=marker_label(tissue_array)
embedding = run_umap(tissue_array)
rgb = assign_colours(embedding)


### for plotting spatial markers
for i in range(len(zarray)):
    rgb = assign_colours(embedding)
    rgb=rgb_marker_img(rgb,marker_binary,i)
    rgb_image = make_rgb_image(rgb, mask)

    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title(allmarker[i])
    # plt.show()
    plt.savefig(allmarker[i]+'.jpg')

## differential analysis: heatmap
rgb, label=kmeans_rgb(embedding, rgb)
allmarker=['DNA','Ki-67','Keratin','CD3D','CD4','CD45','CD8A','α-SMA','CD20']
diff_marker_all=[]
for i in range(3):
    diff_marker=diff(tissue_array,label,i)
    diff_marker_all.append(diff_marker)

diff_marker_all=np.array(diff_marker_all)
diff_marker_all = pd.DataFrame(diff_marker_all)

plt.figure()
sns.heatmap(data=diff_marker_all.T,
            # cmap=sns.diverging_palette(220, 10, sep=80, n=7),
            cmap='BuPu',
            xticklabels=['cluster0','cluster1','cluster2'],
            yticklabels=allmarker,
            )
plt.xticks(fontsize=13)
plt.yticks(fontsize=9)
plt.title(input)
# plt.show()
plt.savefig('hm.jpg')

