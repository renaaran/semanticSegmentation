import re
import os
import glob
import numpy as np
from PIL import Image

regex = re.compile(r'\d+')

BASE_PATH = '/home/CAMPUS/180178991/Pictures/CMP_facade_DB_base/base'

def resize_image():
    for file_name in glob.glob('cmp_b*.jpg'):
        idx = int(regex.findall(file_name)[0])
        img = Image.open(file_name)
        img_256x256 = img.resize((256, 256))
        img_256x256.save(f'{idx}.jpg')

def resize_label():
    for file_name in glob.glob(os.path.join(BASE_PATH, 'cmp_b*.png')):
        print(file_name)
        idx = int(regex.findall(file_name)[0])
        img = Image.open(file_name)
        img_256x256 = img.resize((256, 256))
        img_256x256.save(f'{idx}.png')
        if not np.array_equal(np.unique(np.array(img)),
                              np.unique(np.array(img_256x256))):
            print(np.unique(np.array(img_256x256)))
            print(np.unique(np.array(img)))


if __name__ == '__main__':
    resize_label()