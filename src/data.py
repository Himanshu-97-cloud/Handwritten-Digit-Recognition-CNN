
import keras


def load_data():

    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize images
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape images
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test, y_test
