import umap
import trimap
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetV2L


def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def generate_latent_data() -> pd.DataFrame:
    (_, _), (x_test, y_test) = cifar10.load_data()
    y_test_ohe = to_categorical(y_test)
    model = _build_model("EfficientNetV2L")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    # Train
    model.fit(x_test, y_test_ohe, epochs=50, batch_size=32)

    # Evaluate
    _, acc = model.evaluate(x_test, y_test_ohe)
    print(f"Accuracy: {acc}")

    # Predict using latent features
    model.pop()
    latent_data = model.predict(x_test)
    latent_data = latent_data.reshape(latent_data.shape[0], -1)

    emb_latent_trimap = trimap.TRIMAP().fit_transform(latent_data)
    emb_latent_umap = umap.UMAP().fit_transform(latent_data)
    emb_latent_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(latent_data)

    df_latent = pd.DataFrame({
        'x': emb_latent_trimap[:, 0].squeeze(),
        'y': emb_latent_trimap[:, 1].squeeze(),
        'x_umap': emb_latent_umap[:, 0].squeeze(),
        'y_umap': emb_latent_umap[:, 1].squeeze(),
        'x_tsne': emb_latent_tsne[:, 0].squeeze(),
        'y_tsne': emb_latent_tsne[:, 1].squeeze(),
        'label': y_test.squeeze(),
        'index': np.arange(len(y_test))
    })
    return df_latent


def _build_model(model_name: str):
    if model_name == "EfficientNetV2L":
        return Sequential([
            EfficientNetV2L(input_shape=(32, 32, 3), include_top=False, classes=10),
            Flatten(),
            Dense(10, activation='softmax')
        ])
    else:
        raise ValueError(f"Model {model_name} not found")
