
import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path


# -------------------------
# 1. PAGE CONFIGURATION
# -------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_model.keras"


# -------------------------
# 2. LOAD MODEL
# -------------------------

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)


# -------------------------
# 3. IMAGE PREPROCESSING
# -------------------------

def preprocess_image(image):

    image = Image.fromarray(
        image.astype("uint8")
    ).convert("L")

    image_array = np.array(image)

    # Find digit boundaries
    coords = np.argwhere(image_array > 30)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop digit
    image = image.crop(
        (x_min, y_min, x_max + 1, y_max + 1)
    )

    # Resize to 20x20
    image.thumbnail((20, 20))

    # Create 28x28 black background
    new_image = Image.new("L", (28, 28), 0)

    # Center digit
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    new_image.paste(image, (x_offset, y_offset))

    # Normalize and reshape
    image_array = np.array(new_image).astype("float32") / 255.0

    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


# -------------------------
# 4. STREAMLIT INTERFACE
# -------------------------

st.title("✍️ Handwritten Digit Recognizer")

st.write("Draw a handwritten digit between 0 and 9.")

st.caption("Use your mouse to draw inside the canvas.")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas"
)


# -------------------------
# 5. PREDICTION
# -------------------------

if st.button("Predict Digit", type="primary", use_container_width=True):

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        processed_image = preprocess_image(
            canvas_result.image_data
        )

        if processed_image is None:

            st.warning("Please draw a digit first.")

        elif not MODEL_PATH.exists():

            st.error("Trained model not found. Train the model first.")

        else:

            classifier = load_trained_model()

            prediction = classifier.predict(
                processed_image,
                verbose=0
            )

            predicted_digit = np.argmax(prediction[0])

            confidence = prediction[0][predicted_digit] * 100

            # Display prediction
            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Digit",
                    predicted_digit
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            # Display probabilities
            st.subheader("Prediction Probabilities")

            probabilities = prediction[0]

            # Two horizontal rows of five digits
            for start in range(0, 10, 5):

                cols = st.columns(5)

                for i, col in enumerate(cols):

                    digit = start + i
                    probability = float(probabilities[digit])

                    with col:

                        st.markdown(f"**Digit {digit}**")

                        st.progress(probability)

                        st.caption(
                            f"{probability * 100:.1f}%"
                        )
