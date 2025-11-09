import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, \
    BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image


def load_data(dir_pict, dir_mask, pict_height, pict_width):
    dir_pict_files = sorted(os.listdir(dir_pict))
    dir_mask_files = sorted(os.listdir(dir_mask))

    pict = np.empty((len(dir_pict_files), pict_height, pict_width, 3), dtype=np.float32)
    mask = np.empty((len(dir_mask_files), pict_height, pict_width), dtype=np.uint8)

    for i, pict_filename in enumerate(dir_pict_files):
        pict_filepath = os.path.join(dir_pict, pict_filename)
        pict[i] = img_to_array(load_img(pict_filepath, target_size=(pict_height, pict_width)))
    pict /= 255

    for i, mask_filename in enumerate(dir_mask_files):
        mask_filepath = os.path.join(dir_mask, mask_filename)
        msk = img_to_array(load_img(mask_filepath, target_size=(pict_height, pict_width), color_mode='grayscale'))
        mask[i] = msk.reshape(pict_height, pict_width).astype(np.uint8)

    return pict, mask


def fcn_model(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    bn1 = BatchNormalization()(conv1)
    af1 = Activation('relu')(bn1)
    conv2 = Conv2D(32, 3, padding='same', use_bias=False)(af1)
    bn2 = BatchNormalization()(conv2)
    af2 = Activation('relu')(bn2)
    conv3 = Conv2D(32, 3, padding='same', use_bias=False)(af2)
    bn3 = BatchNormalization()(conv3)
    af3 = Activation('relu')(bn3)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(af3)

    conv4 = Conv2D(64, 3, padding='same', use_bias=False)(pool1)
    bn4 = BatchNormalization()(conv4)
    af4 = Activation('relu')(bn4)
    conv5 = Conv2D(64, 3, padding='same', use_bias=False)(af4)
    bn5 = BatchNormalization()(conv5)
    af5 = Activation('relu')(bn5)
    conv6 = Conv2D(64, 3, padding='same', use_bias=False)(af5)
    bn6 = BatchNormalization()(conv6)
    af6 = Activation('relu')(bn6)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(af6)

    conv7 = Conv2D(128, 3, padding='same', use_bias=False)(pool2)
    bn7 = BatchNormalization()(conv7)
    af7 = Activation('relu')(bn7)
    conv8 = Conv2D(128, 3, padding='same', use_bias=False)(af7)
    bn8 = BatchNormalization()(conv8)
    af8 = Activation('relu')(bn8)
    conv9 = Conv2D(128, 3, padding='same', use_bias=False)(af8)
    bn9 = BatchNormalization()(conv9)
    af9 = Activation('relu')(bn9)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(af9)

    # Bottleneck
    conv10 = Conv2D(256, 3, padding='same', use_bias=False)(pool3)
    bn10 = BatchNormalization()(conv10)
    af10 = Activation('relu')(bn10)
    conv11 = Conv2D(256, 3, padding='same', use_bias=False)(af10)
    bn11 = BatchNormalization()(conv11)
    af11 = Activation('relu')(bn11)
    conv12 = Conv2D(256, 3, padding='same', use_bias=False)(af11)
    bn12 = BatchNormalization()(conv12)
    af12 = Activation('relu')(bn12)

    # Decoder
    up13 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(af12)
    up13 = concatenate([up13, af9], axis=3)
    conv13 = Conv2D(128, 3, padding='same', use_bias=False)(up13)
    bn13 = BatchNormalization()(conv13)
    af13 = Activation('relu')(bn13)

    up14 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(af13)
    up14 = (concatenate([up14, af6], axis=3))
    conv14 = Conv2D(64, 3, padding='same', use_bias=False)(up14)
    bn14 = BatchNormalization()(conv14)
    af14 = Activation('relu')(bn14)

    up15 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(af14)
    up15 = concatenate([up15, af3], axis=3)
    conv15 = Conv2D(32, 3, padding='same', use_bias=False)(up15)
    bn15 = BatchNormalization()(conv15)
    af15 = Activation('relu')(bn15)

    outputs = Conv2D(num_classes, 1, activation='softmax')(af15)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def masks_to_categorical(masks, num_classes):
    num_samples, height, width = masks.shape
    masks_categorical = np.zeros((num_samples, height, width, num_classes), dtype=np.float32)

    for i in range(num_samples):
        for c in range(num_classes):
            masks_categorical[i, :, :, c] = (masks[i] == c).astype(np.float32)

    return masks_categorical


def main():
    preprocessed_pict_path = 'oxford_iiit_pet/preprocessed_pict'
    preprocessed_mask_path = 'oxford_iiit_pet/preprocessed_mask'

    pict_sample = Image.open('oxford_iiit_pet/preprocessed_pict/Abyssinian_1.jpg')
    pict_width, pict_height = pict_sample.size

    print(f"Original dimensions: {pict_width}x{pict_height}")

    pict, mask = load_data(preprocessed_pict_path, preprocessed_mask_path, pict_height, pict_width)

    print(f"Pictures shape: {pict.shape}")
    print(f"Masks shape: {mask.shape}")
    print(f"Picture value range: [{pict.min()}, {pict.max()}]")
    print(f"Mask unique values: {np.unique(mask)}")

    NUM_CLASSES = len(np.unique(mask))
    print(f"Number of classes: {NUM_CLASSES}")

    cat_mask = masks_to_categorical(mask, NUM_CLASSES)

    x_train, x_val, y_train, y_val = train_test_split(pict, cat_mask, test_size=0.2, random_state=42)

    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")


    model = fcn_model((pict_height, pict_width, 3), NUM_CLASSES)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=13,
                        batch_size=40,
                        validation_data=(x_val, y_val))

    loss, accuracy = model.evaluate(x_val, y_val)
    print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')

    model.save('final_model_fcnv2.h5')
    print('Model saved successfully!')


if __name__ == '__main__':
    main()