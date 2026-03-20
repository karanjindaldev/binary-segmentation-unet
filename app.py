import streamlit as st
import tensorflow as tf
from numpy import expand_dims, squeeze, dstack
from keras.models import load_model
from PIL import Image

model = load_model('model_v1.keras')

st.title("Background Remover")

image_upload = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "avif", "webp"])
result_image = None

if image_upload:
    image = Image.open(image_upload)
    resized_image = image.resize((128, 128))

    img_array = tf.keras.utils.img_to_array(resized_image)
    img_array = expand_dims(img_array, axis=0)

    result_array = model.predict(img_array)[0]
    mask = (result_array > 0.5).astype('uint8')
    print('mask shape:', mask.shape)
    result_img_array = dstack((squeeze(img_array), mask*255))
    print('result img array shape: ', result_img_array.shape)
    result_img_array = result_img_array.astype('uint8')

    result_image = Image.fromarray(result_img_array)

with st.container(horizontal=True, horizontal_alignment="center", gap="large"):

    if image_upload is not None:
        st.image(resized_image, width=256, caption="Original Image")

    if result_image is not None:
        st.image(result_image, width=256, caption="Background Removed")