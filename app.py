import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_cnn.keras")

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# App title
# --------------------------------------------------

st.title("Cat vs Dog Classifier")

st.write(
    "Upload an image and let the CNN predict whether it is a Cat or Dog."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    if prediction >= 0.5:
        predicted_class = "Dog 🐶"
        confidence = prediction * 100
    else:
        predicted_class = "Cat 🐱"
        confidence = (1 - prediction) * 100

    st.subheader("Prediction")

    st.success(predicted_class)

    st.write(
        f"Confidence: **{confidence:.2f}%**"
    )