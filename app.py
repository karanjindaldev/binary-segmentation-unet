import streamlit as st
from model import unet
import tensorflow as tf
from PIL import Image
from io import BytesIO
import gdown
import os

url = 'https://drive.google.com/uc?id=146PXTfjwXPa0Z4avylIBZZAaOMZFmH4R'
weights = 'unet.weights.h5'

if not os.path.exists(weights):
    gdown.download(url, weights)

model = unet(input_shape=(128, 128, 3), is_inside_notebook=False)
model.load_weights(weights)

st.title("Background Remover")

image_upload = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "avif", "webp"])
final_image = None

if image_upload:
    uploaded_image = Image.open(image_upload).convert("RGB")
    resized_image = uploaded_image.resize((128, 128))

    image_array = tf.keras.utils.img_to_array(resized_image)
    image_array = tf.expand_dims(image_array, axis=0)

    prediction_array = model.predict(image_array)[0]
    alpha_mask = (prediction_array > 0.5).astype('uint8') * 255

    final_image_array = tf.experimental.numpy.dstack((tf.squeeze(image_array), alpha_mask)).numpy()
    final_image_array = final_image_array.astype('uint8')

    final_image = Image.fromarray(final_image_array)
    download_img = BytesIO()
    final_image.save(download_img, format='png')
    download_img.seek(0)

with st.container(horizontal=True, horizontal_alignment="center", gap="large"):

    if image_upload:
        st.image(resized_image, width=256, caption="Original Image")

    if final_image:
        st.image(final_image, width=256, caption="Background Removed")
        st.download_button(label='Download Image', data=download_img, file_name='result.png', mime='image/png')