import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import minimize
from keras.datasets import mnist
import trimap
import umap
import pacmap
import pandas as pd
from utils import (
    convert_images_to_base64,
    compute_centroids,
    compute_pairwise_distances,
    objective_function,
    compute_translations
)

def load_mnist(percentage=100):
    """
    outputs the mnist needed to train trimap
    used percentage for debugging and not needing to reload the entire time
    """
    (train_examples, train_labels), (test_examples, test_labels) = mnist.load_data()

    # Calculate the number of samples to load
    train_samples = int(len(train_examples) * percentage / 100)
    test_samples = int(len(test_examples) * percentage / 100)

    # Slice the arrays to get the specified percentage of data
    train_examples = train_examples[:train_samples].astype(np.float32)
    train_labels = train_labels[:train_samples]
    test_examples = test_examples[:test_samples].astype(np.float32)
    test_labels = test_labels[:test_samples]
    return train_examples, train_labels, test_examples, test_labels

train_examples, train_labels, test_examples, test_labels = load_mnist()
examples = np.concatenate((train_examples, test_examples), 0)
base64_images = convert_images_to_base64(examples)
labels = np.concatenate((train_labels, test_labels))
indices = np.arange(len(examples))
examples = examples.reshape(examples.shape[0], -1)

np.save('./data/mnist_array', examples)

latent_centroids = compute_centroids(examples, labels)
desired_distances = compute_pairwise_distances(np.array([*latent_centroids.values()]))

df = pd.DataFrame({
        'label': labels,
        'image': base64_images,
        'index': indices
    })

print('\ntrimap\n')
# df = pd.read_pickle('./data/mnist_param_grid_data.pkl')
# trimap
trimap_n_in = [8, 12, 16]
trimap_n_out = [2, 4, 8]
for i in trimap_n_in:
    for j in trimap_n_out:
        emb_mnist_trimap = trimap.TRIMAP(n_inliers=i, n_outliers=j).fit_transform(examples)
        df['x_trimap_nin_'+str(i)+'_nout_'+str(j)] = emb_mnist_trimap[:, 0]
        df['y_trimap_nin_'+str(i)+'_nout_'+str(j)] = emb_mnist_trimap[:, 1]

# import ipdb
# ipdb.set_trace()

df['x_trimap_nin_12_nout_4'] = (df['x_trimap_nin_12_nout_4'] - df['x_trimap_nin_12_nout_4'].min())/(df['x_trimap_nin_12_nout_4'].max() - df['x_trimap_nin_12_nout_4'].min())

df['y_trimap_nin_12_nout_4'] = (df['y_trimap_nin_12_nout_4'] - df['y_trimap_nin_12_nout_4'].min())/(df['y_trimap_nin_12_nout_4'].max() - df['y_trimap_nin_12_nout_4'].min())

trimap_centroids = compute_centroids(np.array(df[['x_trimap_nin_12_nout_4', 'y_trimap_nin_12_nout_4']]), np.array(df['label']))

optimal_positions = minimize(objective_function,
                                np.array([*trimap_centroids.values()]).reshape(20),
                                method='L-BFGS-B',
                                args=(desired_distances,))
translations = compute_translations(trimap_centroids, optimal_positions.x.reshape(10, 2))

df['x_trimap_shift'] = df['label'].map(lambda label: translations[label][0])
df['y_trimap_shift'] = df['label'].map(lambda label: translations[label][1])

print('\numap\n')
# umap
umap_n_neighbors = [5, 15, 45]
umap_min_dist = [0.0, 0.1, 0.5]
for i in umap_n_neighbors:
    for j in umap_min_dist:
        emb_mnist_umap = umap.UMAP(n_neighbors=i, min_dist=j).fit_transform(examples)
        df['x_umap_nneighbors_'+str(i)+'_mindist_'+str(j)] = emb_mnist_umap[:, 0]
        df['y_umap_nneighbors_'+str(i)+'_mindist_'+str(j)] = emb_mnist_umap[:, 1]

df['x_umap_nneighbors_15_mindist_0.1'] = (df['x_umap_nneighbors_15_mindist_0.1'] - df['x_umap_nneighbors_15_mindist_0.1'].min())/(df['x_umap_nneighbors_15_mindist_0.1'].max() - df['x_umap_nneighbors_15_mindist_0.1'].min())

df['y_umap_nneighbors_15_mindist_0.1'] = (df['y_umap_nneighbors_15_mindist_0.1'] - df['y_umap_nneighbors_15_mindist_0.1'].min())/(df['y_umap_nneighbors_15_mindist_0.1'].max() - df['y_umap_nneighbors_15_mindist_0.1'].min())

umap_centroids = compute_centroids(np.array(df[['x_umap_nneighbors_15_mindist_0.1', 'y_umap_nneighbors_15_mindist_0.1']]), np.array(df['label']))

optimal_positions = minimize(objective_function,
                                np.array([*umap_centroids.values()]).reshape(20),
                                method='L-BFGS-B',
                                args=(desired_distances,))
translations = compute_translations(umap_centroids, optimal_positions.x.reshape(10, 2))

df['x_umap_shift'] = df['label'].map(lambda label: translations[label][0])
df['y_umap_shift'] = df['label'].map(lambda label: translations[label][1])

print('\npacmap\n')
# pacmap
pacmap_n_neighbors = [5, 10, 20]
pacmap_init = ['pca', 'random']
for i in pacmap_n_neighbors:
    for j in pacmap_init:
        emb_mnist_pacmap = pacmap.PaCMAP(n_neighbors=i).fit_transform(examples, init=j)
        df['x_pacmap_nneighbors_'+str(i)+'_init_'+j] = emb_mnist_pacmap[:, 0]
        df['y_pacmap_nneighbors_'+str(i)+'_init_'+j] = emb_mnist_pacmap[:, 1]

df['x_pacmap_nneighbors_10_init_pca'] = (df['x_pacmap_nneighbors_10_init_pca'] - df['x_pacmap_nneighbors_10_init_pca'].min())/(df['x_pacmap_nneighbors_10_init_pca'].max() - df['x_pacmap_nneighbors_10_init_pca'].min())

df['y_pacmap_nneighbors_10_init_pca'] = (df['y_pacmap_nneighbors_10_init_pca'] - df['y_pacmap_nneighbors_10_init_pca'].min())/(df['y_pacmap_nneighbors_10_init_pca'].max() - df['y_pacmap_nneighbors_10_init_pca'].min())

pacmap_centroids = compute_centroids(np.array(df[['x_pacmap_nneighbors_10_init_pca', 'y_pacmap_nneighbors_10_init_pca']]), np.array(df['label']))

optimal_positions = minimize(objective_function,
                                np.array([*pacmap_centroids.values()]).reshape(20),
                                method='L-BFGS-B',
                                args=(desired_distances,))
translations = compute_translations(pacmap_centroids, optimal_positions.x.reshape(10, 2))

df['x_pacmap_shift'] = df['label'].map(lambda label: translations[label][0])
df['y_pacmap_shift'] = df['label'].map(lambda label: translations[label][1])

print('\ntsne\n')
# t-sne
pca = PCA(n_components=90)  # Reduce to 50 dimensions (arbitrary choice)
examples = pca.fit_transform(examples)

tsne_perp = [15, 30, 45]
tsne_exa = [6, 12, 24]
for i in tsne_perp:
    for j in tsne_exa:
        emb_mnist_tsne = TSNE(perplexity=i, early_exaggeration=j).fit_transform(examples)
        df['x_tsne_perp_'+str(i)+'_exa_'+str(j)] = emb_mnist_tsne[:, 0]
        df['y_tsne_perp_'+str(i)+'_exa_'+str(j)] = emb_mnist_tsne[:, 1]

df['x_tsne_perp_30_exa_12'] = (df['x_tsne_perp_30_exa_12'] - df['x_tsne_perp_30_exa_12'].min())/(df['x_tsne_perp_30_exa_12'].max() - df['x_tsne_perp_30_exa_12'].min())

df['y_tsne_perp_30_exa_12'] = (df['y_tsne_perp_30_exa_12'] - df['y_tsne_perp_30_exa_12'].min())/(df['y_tsne_perp_30_exa_12'].max() - df['y_tsne_perp_30_exa_12'].min())

tsne_centroids = compute_centroids(np.array(df[['x_tsne_perp_30_exa_12', 'y_tsne_perp_30_exa_12']]), np.array(df['label']))

optimal_positions = minimize(objective_function,
                                np.array([*tsne_centroids.values()]).reshape(20),
                                method='L-BFGS-B',
                                args=(desired_distances,))
translations = compute_translations(tsne_centroids, optimal_positions.x.reshape(10, 2))

df['x_tsne_shift'] = df['label'].map(lambda label: translations[label][0])
df['y_tsne_shift'] = df['label'].map(lambda label: translations[label][1])

df.to_pickle('./data/mnist_param_grid_data.pkl')