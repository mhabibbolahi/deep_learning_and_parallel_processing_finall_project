import numpy as np
from PIL import Image
from matplotlib.image import imread
import os


pick_path = 'oxford_iiit_pet/mask'
mask_path = 'oxford_iiit_pet/pick'
pick_list = os.listdir(pick_path)
mask_list = os.listdir(mask_path)
# find suitable width & height process
pick_widths = []
pick_heights = []
for pick_name in pick_list:
    pick = Image.open(os.path.join(pick_path, pick_name))
    pick_width, pick_height = pick.size
    pick_widths.append(pick_width)
    pick_heights.append(pick_height)

print(f'suitable width : {np.mean(pick_heights)},suitable height: {np.mean(pick_widths)}')

pick = []
mask = []
INPUT_HEIGHT = 224
INPUT_WIDTH = 224

for x in pick_list:
    y = Image.open('oxford_iiit_pet/pick/'+x).crop((0, 0, INPUT_WIDTH, INPUT_WIDTH))
    y.save('oxford_iiit_pet/pick2/'+x)
print('55')
for x in pick_list:
    y = Image.open('oxford_iiit_pet/mask/' + x).crop((0, 0, INPUT_WIDTH, INPUT_WIDTH))
    y.save('oxford_iiit_pet/mask2/' + x)
print('55')
