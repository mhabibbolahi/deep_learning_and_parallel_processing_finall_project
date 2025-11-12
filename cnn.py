import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # فقط error و fatal نشون داده می‌شن

warnings.filterwarnings("ignore")          # هشدارهای پایتون رو هم حذف کن
tf.get_logger().setLevel('ERROR')  # فقط error نشون میده، info (مثل epoch و loss) باقی می‌مونه

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

image_size = 32  # CIFAR10 images are 32x32 already

# --- Augmentation pipeline for training data ---
# --- Augmentation pipeline (compatible with TF 2.10) ---
train_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),  # ✅ available
    layers.RandomRotation(0.083),  # ✅ ~15 degrees
    layers.RandomContrast(0.2),  # ✅ available
    layers.RandomTranslation(0.1, 0.1),  # ✅ available
    layers.Rescaling(1. / 255),  # ✅ ToTensor() equivalent
])

train_images, test_images = train_images / 255, test_images / 255

train_labels, test_labels = to_categorical(train_labels, 10), to_categorical(test_labels, 10)

model = models.Sequential([
    train_augmentation,
    layers.Conv2D(32, (3, 3), use_bias=False, padding='same', input_shape=(32, 32, 3),
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.4),

    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.2),

    layers.Dense(10, activation='softmax'),
])
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr=1e-6
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=60,
                    validation_data=(test_images, test_labels), batch_size=64, callbacks=[reduce_lr])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

model.save('model_cnn.h5')
