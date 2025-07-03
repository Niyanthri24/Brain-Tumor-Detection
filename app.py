import streamlit as st # type: ignore
import numpy as np
import cv2 # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image # type: ignore

model = load_model("brain_tumor_detector.h5")
img_size = 150

st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an MRI scan to check for brain tumors.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI.", use_column_width=True)

    if st.button("Predict"):
        img = np.array(image.resize((img_size, img_size)))
        img = img / 255.0
        img = img.reshape(1, img_size, img_size, 3)
        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            st.error("⚠️ Tumor Detected")
        else:
            st.success("✅ No Tumor Detected")
