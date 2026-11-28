
import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path


# Page setup
st.set_page_config(
    page_title="DigitAI - Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "models" / "mnist_model.keras"


# Styling
st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f1f5f9;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

footer {
    visibility: hidden;
}

.brand {
    font-size: 24px;
    font-weight: 800;
    color: white;
}

.brand span {
    color: #818cf8;
}

.nav-tag {
    color: #a5b4fc;
    background: #202a46;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 12px;
}

.hero {
    text-align: center;
    margin: 45px 0 35px;
}

.hero h1 {
    color: white;
    font-size: clamp(30px, 5vw, 46px);
    font-weight: 800;
}

.hero p {
    color: #94a3b8;
    font-size: 15px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
}

.section-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 18px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #151d32;
    border: 1px solid #293650;
    border-radius: 15px;
    padding: 16px;
}

.stButton > button {
    background: #6366f1;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #4f46e5;
    color: white;
    border: none;
}

.result-card {
    background: #202a46;
    border: 1px solid #394768;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 15px 0;
}

.result-number {
    color: white;
    font-size: 82px;
    font-weight: 800;
}

.result-label {
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 600;
}

.custom-progress {
    height: 9px;
    width: 100%;
    background: #293650;
    border-radius: 20px;
    overflow: hidden;
    margin: 7px 0;
}

.custom-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 20px;
}

.progress-label {
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 700;
    text-align: right;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 45px;
    padding: 20px;
    border-top: 1px solid #26324b;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .hero {
        margin: 30px 0;
    }
}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Prepare drawing for CNN
def preprocess_image(image):

    image = Image.fromarray(
        image.astype("uint8")
    ).convert("L")

    array = np.array(image)
    coords = np.argwhere(array > 30)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    image = image.crop(
        (x_min, y_min, x_max + 1, y_max + 1)
    )

    image.thumbnail((20, 20))

    centered = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    centered.paste(image, (x, y))

    array = np.array(centered).astype("float32") / 255.0

    return array.reshape(1, 28, 28, 1)


# Navigation
nav1, nav2 = st.columns([3, 1])

with nav1:
    st.markdown(
        '<div class="brand">✍️ Digit<span>AI</span></div>',
        unsafe_allow_html=True
    )

with nav2:
    st.markdown(
        '<div class="nav-tag">CNN · DEEP LEARNING</div>',
        unsafe_allow_html=True
    )


# Heading
st.markdown("""
<div class="hero">
    <h1>Handwritten Digit Recognition</h1>
    <p>Draw a digit from 0 to 9 and let our CNN model recognize it.</p>
</div>
""", unsafe_allow_html=True)


# Store prediction
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# Drawing and result sections
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Draw Your Digit</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">Use your mouse or touchscreen.</div>',
            unsafe_allow_html=True
        )

        canvas = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            return_image_data=True,
            key="digit_canvas"
        )

        predict = st.button(
            "✦ Recognize Digit",
            type="primary",
            use_container_width=True
        )


with right:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Recognition Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">Your prediction appears here.</div>',
            unsafe_allow_html=True
        )

        if predict:

            if canvas.image_data is None:
                st.warning("Please draw a digit first.")

            else:
                processed = preprocess_image(canvas.image_data)

                if processed is None:
                    st.warning("Please draw a digit first.")

                elif not MODEL_PATH.exists():
                    st.error("Model not found. Train your model first.")

                else:
                    try:
                        with st.spinner("Recognizing digit..."):
                            model = load_model()
                            result = model.predict(processed, verbose=0)
                            st.session_state.prediction = result[0]

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

        if st.session_state.prediction is not None:

            probabilities = st.session_state.prediction
            digit = int(np.argmax(probabilities))
            confidence = float(probabilities[digit]) * 100

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">PREDICTED DIGIT</div>
                <div class="result-number">{digit}</div>
                <div style="color:#94a3b8">{confidence:.2f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Model Confidence**")

            st.markdown(f"""
            <div class="custom-progress">
                <div class="custom-progress-fill"
                     style="width:{confidence:.4f}%"></div>
            </div>
            <div class="progress-label">{confidence:.2f}%</div>
            """, unsafe_allow_html=True)

        else:
            st.info("Draw a digit and click Recognize.")


# Probability comparison
if st.session_state.prediction is not None:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Prediction Breakdown</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">Probability for each digit (0–9).</div>',
            unsafe_allow_html=True
        )

        probabilities = st.session_state.prediction

        for digit in np.argsort(probabilities)[::-1]:

            percentage = float(probabilities[digit]) * 100

            col1, col2, col3 = st.columns([0.7, 3, 0.8])

            with col1:
                st.markdown(f"**{digit}**")

            with col2:
                st.markdown(f"""
                <div class="custom-progress">
                    <div class="custom-progress-fill"
                         style="width:{percentage:.4f}%"></div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(
                    f'<div class="progress-label">{percentage:.2f}%</div>',
                    unsafe_allow_html=True
                )


# Footer
st.markdown("""
<div class="footer">
    Built with Python · TensorFlow · Keras · Streamlit
    <br><br>
    Handwritten Digit Recognition Using CNN
</div>
""", unsafe_allow_html=True)
