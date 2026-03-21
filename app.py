import streamlit as st
import tensorflow as tf
from numpy import expand_dims, squeeze, dstack
from keras.models import load_model
from PIL import Image
from io import BytesIO

model = load_model('model_v1.keras')

st.title("Background Remover")

image_upload = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "avif", "webp"])
result_image = None

if image_upload:
    image = Image.open(image_upload).convert("RGB")
    resized_image = image.resize((128, 128))

    img_array = tf.keras.utils.img_to_array(resized_image)
    img_array = expand_dims(img_array, axis=0)

    result_array = model.predict(img_array)[0]
    mask = (result_array > 0.5).astype('uint8')

    result_img_array = dstack((squeeze(img_array), mask*255))
    result_img_array = result_img_array.astype('uint8')

    result_image = Image.fromarray(result_img_array)
    download_img = BytesIO()
    result_image.save(download_img, format='png')
    download_img.seek(0)

with st.container(horizontal=True, horizontal_alignment="center", gap="large"):

    if image_upload is not None:
        st.image(resized_image, width=256, caption="Original Image")

    if result_image is not None:
        st.image(result_image, width=256, caption="Background Removed")
        st.download_button(label='Download Image', data=download_img, file_name='result.png', mime='image/png')