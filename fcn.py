import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image


def load_data(dir_pict, dir_mask):
    dir_pict_files = os.listdir(dir_pict)
    dir_mask_files = os.listdir(dir_mask)
    pict = np.empty((len(dir_pict_files), pict_width, pict_height, 3), dtype=np.float16)
    mask = np.empty((len(dir_mask_files), pict_width, pict_height), dtype=np.float16)

    for i, pict_filename in enumerate(dir_pict_files):
        pict_filepath = os.path.join(dir_pict, pict_filename)
        pict[i] = img_to_array(load_img(pict_filepath, target_size=(pict_width, pict_height)))

    for i, mask_filename in enumerate(dir_mask_files):
        mask_filepath = os.path.join(dir_mask, mask_filename)
        mask[i] = img_to_array(load_img(mask_filepath, target_size=(pict_width, pict_height), color_mode='grayscale')).reshape(pict_width, pict_height)

    return pict, mask


def fcn_model(input_shape, num_classes):
    inputs = Input(input_shape)

    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)

    up5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = concatenate([up5, conv3], axis=3)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(up5)

    up6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(up6)

    up7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(up7)

    outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    preprocessed_pict_path = 'oxford_iiit_pet/preprocessed_pict'
    preprocessed_mask_path = 'oxford_iiit_pet/preprocessed_mask'
    dir_pict_files = os.listdir(preprocessed_mask_path)

    pict = Image.open(os.path.join(preprocessed_pict_path, dir_pict_files[0]))
    pict_width, pict_height = pict.size

    pict, mask = load_data(preprocessed_pict_path, preprocessed_mask_path)
    x_train, x_val, y_train, y_val = train_test_split(pict, mask, test_size=0.2, random_state=42)
    NUM_CLASSES = len(np.unique(y_train))
    print(NUM_CLASSES)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

    model = fcn_model((pict_width, pict_height, 3), NUM_CLASSES)

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    loss, accuracy = model.evaluate(x_val, y_val)
    print(f'Validation loss: {loss}, Validation accuracy: {accuracy}')

    model.save('final_model_fcn.h5')

    print('model saved')

if __name__ == '__main__':
    main()