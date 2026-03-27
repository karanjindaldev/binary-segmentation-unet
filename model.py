import tensorflow as tf

def unet(input_shape, filters, is_inside_notebook):
    input_ = tf.keras.Input(shape=input_shape)
    rescaled_input = tf.keras.layers.Rescaling(scale=1/255.)(input_)

    skip1, enc1 = encoder_block(rescaled_input, filters)
    skip2, enc2 = encoder_block(enc1, filters*2)
    skip3, enc3 = encoder_block(enc2, filters*4)
    skip4, enc4 = encoder_block(enc3, filters*8)

    enc_pen_ultimate = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he')(enc4)
    enc_ultimate = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu',kernel_initializer='he')(enc_pen_ultimate)

    dec1 = decoder_block(enc_ultimate, skip4, filters*8)
    dec2 = decoder_block(dec1, skip3, filters*4)
    dec3 = decoder_block(dec2, skip2, filters*2)
    dec4 = decoder_block(dec3, skip1, filters)

    final_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(dec4)

    model = tf.keras.Model(inputs=[input_], outputs=[final_output])

    if is_inside_notebook:
        model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        model.summary()
        
    return model

def encoder_block(input_, filters):
    out1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he')(input_)
    out2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he')(out1)
    output = tf.keras.layers.MaxPool2D(strides=(2, 2))(out2)
    return out2, output

def decoder_block(input_, skip, filters):
    out1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he')(input_)
    out2 = tf.keras.layers.concatenate([skip, out1], axis=-1)
    out3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he')(out2)
    output = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he')(out3)
    return output

if __name__ == "__main__":
    model = unet(input_shape=(128, 128, 3))
    print("Execution successful.")