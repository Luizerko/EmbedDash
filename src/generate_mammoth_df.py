import trimap
import umap
import pacmap
import numpy as np
import pandas as pd
from utils import load_mammoth
from sklearn.manifold import TSNE

df_mammoth = load_mammoth()

print('\ntrimap\n')
# df_mammoth = pd.read_pickle('./data/mnist_param_grid_data.pkl')
# trimap
trimap_n_in = [8, 12, 16]
trimap_n_out = [2, 4, 8]
for i in trimap_n_in:
    for j in trimap_n_out:
        emb_mnist_trimap = trimap.TRIMAP(n_dims=3, n_inliers=i, n_outliers=j).fit_transform(df_mammoth[['x', 'y', 'z']].to_numpy())
        df_mammoth['x_trimap_nin_'+str(i)+'_nout_'+str(j)] = emb_mnist_trimap[:, 0]
        df_mammoth['y_trimap_nin_'+str(i)+'_nout_'+str(j)] = emb_mnist_trimap[:, 1]
        df_mammoth['z_trimap_nin_'+str(i)+'_nout_'+str(j)] = emb_mnist_trimap[:, 2]

# import ipdb
# ipdb.set_trace()

print('\numap\n')
# umap
umap_n_neighbors = [5, 15, 45]
umap_min_dist = [0.0, 0.1, 0.5]
for i in umap_n_neighbors:
    for j in umap_min_dist:
        emb_mnist_umap = umap.UMAP(n_components=3, n_neighbors=i, min_dist=j).fit_transform(df_mammoth[['x', 'y', 'z']].to_numpy())
        df_mammoth['x_umap_nneighbors_'+str(i)+'_mindist_'+str(j)] = emb_mnist_umap[:, 0]
        df_mammoth['y_umap_nneighbors_'+str(i)+'_mindist_'+str(j)] = emb_mnist_umap[:, 1]
        df_mammoth['z_umap_nneighbors_'+str(i)+'_mindist_'+str(j)] = emb_mnist_umap[:, 2]

print('\npacmap\n')
# pacmap
pacmap_n_neighbors = [5, 10, 20]
pacmap_init = ['pca', 'random']
for i in pacmap_n_neighbors:
    for j in pacmap_init:
        emb_mnist_pacmap = pacmap.PaCMAP(n_components=3, n_neighbors=i).fit_transform(df_mammoth[['x', 'y', 'z']].to_numpy(), init=j)
        df_mammoth['x_pacmap_nneighbors_'+str(i)+'_init_'+j] = emb_mnist_pacmap[:, 0]
        df_mammoth['y_pacmap_nneighbors_'+str(i)+'_init_'+j] = emb_mnist_pacmap[:, 1]
        df_mammoth['z_pacmap_nneighbors_'+str(i)+'_init_'+j] = emb_mnist_pacmap[:, 2]

print('\ntsne\n')
# t-sne
tsne_perp = [15, 30, 45]
tsne_exa = [6, 12, 24]
for i in tsne_perp:
    for j in tsne_exa:
        emb_mnist_tsne = TSNE(n_components=3, perplexity=i, early_exaggeration=j).fit_transform(df_mammoth[['x', 'y', 'z']].to_numpy())
        df_mammoth['x_tsne_perp_'+str(i)+'_exa_'+str(j)] = emb_mnist_tsne[:, 0]
        df_mammoth['y_tsne_perp_'+str(i)+'_exa_'+str(j)] = emb_mnist_tsne[:, 1]
        df_mammoth['z_tsne_perp_'+str(i)+'_exa_'+str(j)] = emb_mnist_tsne[:, 2]

df_mammoth.to_pickle('./data/mammoth_param_grid_data.pkl')