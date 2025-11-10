import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array



def preprocess_input_image(model, image_path):
    input_shape = model.input_shape
    target_size = input_shape[1:3]
    image = load_img(image_path, target_size=target_size)
    image_size = Image.open(image_path).size
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image_size

def main(input_image_path):
    model = load_model('final_model_continued.h5', compile=False)
    label_colors = np.random.randint(0, 255, (40, 3), dtype=np.uint8)
    input_image, image_size = preprocess_input_image(model, input_image_path)

    mask = model.predict(input_image)
    mask = np.squeeze(mask, axis=0)
    predicted_mask = np.argmax(mask, axis=-1)
    output_mask = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
    for label in range(len(label_colors)):
        output_mask[predicted_mask == label] = label_colors[label]
    mask_image = Image.fromarray(output_mask)
    mask_image = mask_image.resize(image_size)
    mask_image.save(input_image_path)

    return input_image_path


