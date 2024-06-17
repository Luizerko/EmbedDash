def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def predict_latent(model, data):
    # remove the last layer from the model
    # model.layers.pop()
    # predict the images
    predictions = model.predict(data)

    return predictions

