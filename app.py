import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error(
        "Please install streamlit-drawable-canvas:\n\n"
        "pip install streamlit-drawable-canvas"
    )
    st.stop()


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_mnist_model():
    return load_model("mnist.h5")


model = load_mnist_model()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================
def preprocess_image(img):

    img = img.convert("L")
    img = np.array(img)

    # Convert black digit on white background
    # to white digit on black background
    img = 255 - img

    coords = np.argwhere(img > 30)

    if len(coords) == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    img = img[y0:y1 + 1, x0:x1 + 1]

    h, w = img.shape

    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))

    img = Image.fromarray(img)
    img = img.resize((new_w, new_h))

    img = np.array(img)

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = img

    processed = canvas.astype("float32") / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    return processed, canvas


# ==========================================
# PREDICTION
# ==========================================
def predict_digit(img):

    result = preprocess_image(img)

    if result is None:
        return None

    processed, debug_img = result

    pred = model.predict(processed, verbose=0)[0]

    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return digit, confidence, pred, debug_img


# ==========================================
# UI
# ==========================================
st.title("✍️ Handwritten Digit Recognition")

st.markdown(
    """
Draw a digit in the canvas and click **Predict**.

The CNN model was trained on the MNIST dataset and can recognize digits from **0–9**.
"""
)

# ==========================================
# DRAWING CANVAS
# ==========================================
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="black",
    background_color="white",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas",
)

# ==========================================
# PREDICT BUTTON
# ==========================================
if st.button("🔍 Predict", use_container_width=True):

    if canvas_result.image_data is not None:

        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        result = predict_digit(img)

        if result is not None:

            digit, confidence, probs, debug_img = result

            st.divider()

            col1, col2 = st.columns([1, 1])

            # ------------------------
            # Prediction Section
            # ------------------------
            with col1:

                st.subheader("Prediction")

                st.success(
                    f"Predicted Digit: {digit}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            # ------------------------
            # Processed Image
            # ------------------------
            with col2:

                st.subheader("Processed 28×28 Image")

                st.image(
                    debug_img,
                    width=220,
                    clamp=True
                )

            st.divider()

            # ------------------------
            # Probabilities
            # ------------------------
            st.subheader("Prediction Probabilities")

            for i, p in enumerate(probs):

                st.markdown(
                    f"**Digit {i}: {p * 100:.2f}%**"
                )

                st.progress(float(p))

        else:
            st.warning(
                "Please draw a digit before predicting."
            )