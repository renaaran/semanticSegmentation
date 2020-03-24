import re
import os
import glob
import numpy as np
from PIL import Image

regex = re.compile(r'\d+')

BASE_PATH = '/home/CAMPUS/180178991/Pictures/CMP_facade_DB_extended/extended'

def resize(mask):
    for file_name in sorted(glob.glob(os.path.join(BASE_PATH, mask))):
        print(file_name)
        idx = int(regex.findall(os.path.basename(file_name))[0])
        img = Image.open(file_name)
        img_256x256 = img.resize((256, 256))
        yield img, img_256x256, idx

def resize_image():
    for _, img_256x256, idx in resize('cmp_x*.jpg'):
        img_256x256.save(os.path.join(BASE_PATH, f'{idx}.jpg'))

def resize_label():
    for img, img_256x256, idx in resize('cmp_x*.png'):
        img_256x256.save(os.path.join(BASE_PATH, f'{idx}.png'))
        if not np.array_equal(np.unique(np.array(img)),
                              np.unique(np.array(img_256x256))):
            print(np.unique(np.array(img_256x256)))
            print(np.unique(np.array(img)))


if __name__ == '__main__':
    resize_image()
    resize_label()