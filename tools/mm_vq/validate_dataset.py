import glob
import mmcv
import numpy as np

folder = 'X:\mapgpt\masks_png'
img_files = glob.glob(folder + '/*.png')
for img_file in img_files:
    print(img_file)
    img = mmcv.imread(img_file, 'unchanged')
    # calculate the number of unique colors in the image
    unique_colors = np.bincount(img.flatten())
    if len(unique_colors) > 6:
        print(unique_colors)
        print('Error: more than 6 unique colors in image')
        continue