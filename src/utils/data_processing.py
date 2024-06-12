import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist


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

def load_mammoth():
    df = pd.read_json("data/mammoth_3d_50k.json")
    df = df.rename(columns={0: 'x', 1: 'y', 2: 'z'})
    return df

def convert_image_to_base64(img):
    buf = io.BytesIO()
    plt.imsave(buf, img.reshape((img.shape[0]**2, 1)), format='png', cmap='gray')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def convert_images_to_base64(images):
    base64_images = np.array([convert_image_to_base64(img) for img in images])
    return base64_images
