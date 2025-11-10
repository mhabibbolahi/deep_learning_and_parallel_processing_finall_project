import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from PIL import Image
import datetime


def load_data(dir_pict, dir_mask, pict_height, pict_width):
    dir_pict_files = sorted(os.listdir(dir_pict))
    dir_mask_files = sorted(os.listdir(dir_mask))

    print(f"Loading {len(dir_pict_files)} images...")

    pict = np.empty((len(dir_pict_files), pict_height, pict_width, 3), dtype=np.float32)
    mask = np.empty((len(dir_mask_files), pict_height, pict_width), dtype=np.uint8)

    for i, pict_filename in enumerate(dir_pict_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(dir_pict_files)}")
        pict_filepath = os.path.join(dir_pict, pict_filename)
        pict[i] = img_to_array(load_img(pict_filepath, target_size=(pict_height, pict_width)))
    pict /= 255.0

    for i, mask_filename in enumerate(dir_mask_files):
        mask_filepath = os.path.join(dir_mask, mask_filename)
        msk = img_to_array(load_img(mask_filepath, target_size=(pict_height, pict_width), color_mode='grayscale'))
        mask[i] = msk.reshape(pict_height, pict_width).astype(np.uint8)

    print("Data loading completed!")
    return pict, mask


def masks_to_categorical(masks, num_classes):
    num_samples, height, width = masks.shape
    masks_categorical = np.zeros((num_samples, height, width, num_classes), dtype=np.float32)

    print("Converting masks to categorical...")
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_samples}")
        for c in range(num_classes):
            masks_categorical[i, :, :, c] = (masks[i] == c).astype(np.float32)

    return masks_categorical


def continue_training_simple():
    print("=" * 60)
    print("Continuing Training (Simple Mode)")
    print("=" * 60)

    MODEL_PATH = 'final_model_fcnv2.h5'

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return -1

    preprocessed_pict_path = 'oxford_iiit_pet/preprocessed_pict'
    preprocessed_mask_path = 'oxford_iiit_pet/preprocessed_mask'

    pict_sample = Image.open('oxford_iiit_pet/preprocessed_pict/Abyssinian_1.jpg')
    pict_width, pict_height = pict_sample.size

    print(f"Image dimensions: {pict_width}x{pict_height}")

    pict, mask = load_data(preprocessed_pict_path, preprocessed_mask_path, pict_height, pict_width)

    NUM_CLASSES = len(np.unique(mask))
    print(f"Number of classes: {NUM_CLASSES}")

    cat_mask = masks_to_categorical(mask, NUM_CLASSES)
    x_train, x_val, y_train, y_val = train_test_split(pict, cat_mask, test_size=0.2, random_state=42)

    print(f"\nLoading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")

    print("\nEvaluating before fine-tuning...")
    initial_results = model.evaluate(x_val, y_val, verbose=1)
    print(f"Initial: Loss={initial_results[0]:.4f}, Accuracy={initial_results[1]:.4f}")

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='categorical_crossentropy',
        metrics=['accuracy', MeanIoU(num_classes=NUM_CLASSES)]
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model_continued.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1
        )
    ]

    print(f"\nContinuing training for 20 more epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=16,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating after fine-tuning...")
    final_results = model.evaluate(x_val, y_val, verbose=1)

    print("\n" + "=" * 60)
    print(f"Initial: Loss={initial_results[0]:.4f}, Accuracy={initial_results[1]:.4f}")
    print(f"Final:   Loss={final_results[0]:.4f}, Accuracy={final_results[1]:.4f}")
    print(f"Improvement: {(final_results[1] - initial_results[1]) * 100:.2f}%")
    print("=" * 60)

    model.save('final_model_continued.h5')
    print('\nModel saved!')

    return model, history


if __name__ == '__main__':
    model, history = continue_training_simple()
