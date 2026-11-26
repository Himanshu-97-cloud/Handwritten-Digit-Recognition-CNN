
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Dropout


def create_model():

    classifier = Sequential()

    classifier.add(Input(shape=(28, 28, 1)))

    classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=10, activation='softmax'))

    classifier.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return classifier
