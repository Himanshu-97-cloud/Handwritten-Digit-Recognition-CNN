
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from pathlib import Path

from src.data import load_data
from src.model import create_model


# Folder paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "mnist_model.keras"


# Load dataset
X_train, y_train, X_test, y_test = load_data()


# -------------------------
# 1. Training
# -------------------------

def train_model():

    classifier = create_model()

    classifier.summary()

    history = classifier.fit(
        X_train,
        y_train,
        epochs=10,
        validation_split=0.2
    )

    # Save model
    classifier.save(MODEL_PATH)

    print("Model saved successfully!")

    # Plot accuracy
    plt.figure(figsize=(10, 5))

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')

    plt.legend()
    plt.savefig(RESULT_DIR / "accuracy.png")
    plt.show()


# -------------------------
# 2. Evaluation
# -------------------------

def evaluate_model():

    classifier = load_model(MODEL_PATH)

    loss, accuracy = classifier.evaluate(X_test, y_test)

    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Predictions
    y_pred = classifier.predict(X_test)

    y_pred_labels = [np.argmax(i) for i in y_pred]

    # Confusion matrix
    cm = tf.math.confusion_matrix(
        labels=y_test,
        predictions=y_pred_labels
    )

    import seaborn as sns

    plt.figure(figsize=(10, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='crest')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig(RESULT_DIR / "confusion_matrix.png")
    plt.show()


# -------------------------
# 3. Prediction
# -------------------------

def predict_digit(index=2):

    classifier = load_model(MODEL_PATH)

    y_pred = classifier.predict(X_test)

    y_pred_labels = [np.argmax(i) for i in y_pred]

    plt.imshow(X_test[index].reshape(28, 28), cmap="gray")

    plt.title(
        f"Actual: {y_test[index]} | Predicted: {y_pred_labels[index]}"
    )

    plt.axis("off")
    plt.show()


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    train_model()
