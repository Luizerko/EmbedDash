import umap
import trimap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE

from keras.api.applications import EfficientNetV2L
from keras.api.utils import to_categorical
from keras.api.datasets import cifar10
from keras.api.layers import Flatten, Dense
from keras.api.models import Sequential


data_path = Path("data/")
latent_data_path = data_path / "latent_data.pkl"


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train_ohe = to_categorical(y_train)
    y_test_ohe = to_categorical(y_test)
    base_model = EfficientNetV2L(input_shape=(32, 32, 3), include_top=False, classes=10)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(y_train_ohe.shape[1], activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    # Train
    model.fit(x_train, y_train_ohe, epochs=30, batch_size=32)

    # Evaluate
    _, acc = model.evaluate(x_test, y_test_ohe)
    print(f"Accuracy: {acc}")

    # Predict using latent features
    model.pop()
    latent_data = model.predict(x_test)
    print("latent_data.shape:", latent_data.shape)
    latent_data = latent_data.reshape(latent_data.shape[0], -1)
    print("latent_data.shape after reshape:", latent_data.shape)

    emb_latent_trimap = trimap.TRIMAP().fit_transform(latent_data)
    emb_latent_umap = umap.UMAP().fit_transform(latent_data)
    emb_latent_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(latent_data)

    df_latent = pd.DataFrame({ # squeeze the columns
        'x': emb_latent_trimap[:, 0].squeeze(),
        'y': emb_latent_trimap[:, 1].squeeze(),
        'x_umap': emb_latent_umap[:, 0].squeeze(),
        'y_umap': emb_latent_umap[:, 1].squeeze(),
        'x_tsne': emb_latent_tsne[:, 0].squeeze(),
        'y_tsne': emb_latent_tsne[:, 1].squeeze(),
        'label': y_test.squeeze(),
        'index': np.arange(len(y_test))
    })

    df_latent.to_pickle(latent_data_path)
