import glob
import os
import random

folder = r'E:\mapgpt\masks_png'
out_folder = r'E:\mapgpt'
os.makedirs(out_folder + '/train', exist_ok=True)
os.makedirs(out_folder + '/val', exist_ok=True)
files = glob.glob(folder + '/*.png')
random.shuffle(files)
train_files = files[:int(len(files) * 0.8)]
val_files = files[int(len(files) * 0.8):]
for f in train_files:
    os.rename(f, out_folder + '/train/' + os.path.basename(f))
for f in val_files:
    os.rename(f, out_folder + '/val/' + os.path.basename(f))

