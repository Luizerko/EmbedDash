import trimap
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from keras.datasets import mnist


def load_mnist():
  (train_examples, train_labels), (test_examples, test_labels) = mnist.load_data()
  train_examples = np.reshape(
      train_examples, (train_examples.shape[0], -1)).astype(np.float32)
  test_examples = np.reshape(
      test_examples, (test_examples.shape[0], -1)).astype(np.float32)
  return train_examples, train_labels, test_examples, test_labels

# print("stuff imported, next: load MNIST")

# load MNSIT dataset
train_examples, train_labels, test_examples, test_labels = load_mnist()
examples = np.concatenate((train_examples, test_examples), 0)
labels = np.concatenate((train_labels, test_labels))

# print("MNIST loaded, next: embed MNIST")

# (takes a while to load)
emb_mnist_trimap = trimap.TRIMAP().fit_transform(examples)


# print("MNIST embedded, next: plot figure")

cmap = cm.get_cmap('tab10')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cols = cmap(np.linspace(0., 1., len(class_names)))[:, :-1]
colors = cols[labels]
num_methods = 1
figsize=20

plt.figure()
plt.scatter(emb_mnist_trimap[:,0], emb_mnist_trimap[:,1], c=colors, alpha=0.8, s=0.1)

plt.xticks([])
plt.yticks([])
plt.title("TRIMAP embeddings on MNIST")

legend_elements = []
for label, name in enumerate(class_names):
    legend_elements.append(Line2D(
        [0], [0], marker='o', lw=0, markersize=10,
        color=cols[label], label=name, alpha=0.8))
plt.legend(handles=legend_elements)

plt.show()

# print("Figure plotted, end")