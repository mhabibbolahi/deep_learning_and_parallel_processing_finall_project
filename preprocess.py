import numpy as np
from PIL import Image
from matplotlib.image import imread
import os


pict_path = 'oxford_iiit_pet/pict'
mask_path = 'oxford_iiit_pet/mask'
pict_list = os.listdir(pict_path)
mask_list = os.listdir(mask_path)
# find suitable width & height process
pict_widths = []
pict_heights = []
for pict_name in pict_list:
    pict = Image.open(os.path.join(pict_path, pict_name))
    pict_width, pict_height = pict.size
    pict_widths.append(pict_width)
    pict_heights.append(pict_height)

print(f'suitable width : {np.mean(pict_heights)},suitable height: {np.mean(pict_widths)}')

# resize image to make them similar

preprocessed_pict_path = 'oxford_iiit_pet/preprocessed_pict'
preprocessed_mask_path = 'oxford_iiit_pet/preprocessed_mask'

os.makedirs(preprocessed_pict_path, exist_ok=True)
os.makedirs(preprocessed_mask_path, exist_ok=True)

INPUT_WIDTH = int(np.mean(pict_widths))
INPUT_HEIGHT = int(np.mean(pict_heights))

for pict_name in pict_list:
    pict = Image.open(os.path.join(pict_path, pict_name))
    pict = pict.convert('RGB')
    pict = pict.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
    pict.save(os.path.join(preprocessed_pict_path, pict_name), format='JPEG', quality=95)

print('pictures are preprocessed')

for mack_name in mask_list:
    mask = Image.open(os.path.join(mask_path, mack_name))
    mask = mask.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.NEAREST)
    mask.save(os.path.join(preprocessed_mask_path, mack_name))

print('masks are also pre-processed')
