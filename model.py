import tensorflow as tf

def unet(input_shape, is_inside_notebook):
    input_ = tf.keras.Input(shape=input_shape)
    rescaled_input = tf.keras.layers.Rescaling(scale=1/255.)(input_)

    skip1, enc1 = encoder_block(rescaled_input, 8)
    skip2, enc2 = encoder_block(enc1, 16)
    skip3, enc3 = encoder_block(enc2, 32)
    skip4, enc4 = encoder_block(enc3, 64)

    enc_pen_ultimate = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(enc4)
    enc_ultimate = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(enc_pen_ultimate)

    dec1 = decoder_block(enc_ultimate, skip4, 64)
    dec2 = decoder_block(dec1, skip3, 32)
    dec3 = decoder_block(dec2, skip2, 16)
    dec4 = decoder_block(dec3, skip1, 8)

    final_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(dec4)

    model = tf.keras.Model(inputs=[input_], outputs=[final_output])

    if is_inside_notebook:
        model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        model.summary()
        
    return model

def encoder_block(input_, filters):
    out1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(input_)
    out2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(out1)
    output = tf.keras.layers.MaxPool2D(strides=(2, 2))(out2)
    return out2, output

def decoder_block(input_, skip, filters):
    out1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_)
    out2 = tf.keras.layers.concatenate([skip, out1], axis=-1)
    out3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(out2)
    output = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(out3)
    return output

if __name__ == "__main__":
    model = unet(input_shape=(128, 128, 3))
    print("Execution successful.")