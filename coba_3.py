import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("2208108010089_Naila Ghina Rania_Laporan 4.pt")  # Sesuaikan path model YOLO
    classifier = tf.keras.models.load_model("my_jam_model.h5")  # Sesuaikan path model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("YOLO Detection + Image Classification")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    # YOLO Detection
    results = yolo_model.predict(np.array(img))
    st.write("Deteksi YOLO:")
    st.write(results[0].boxes.data)

    # Classification
    img_resized = img.resize((224, 224))  # Sesuaikan dengan input model
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = classifier.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write("Hasil Klasifikasi:", predicted_class)


