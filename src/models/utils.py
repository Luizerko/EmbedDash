def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_and_predict_latent(model, data, labels):
    model.fit(data, labels, epochs=10, batch_size=32)
    model.pop()
    model.pop()
    predictions = model.predict(data)
    return predictions
