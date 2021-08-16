import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def visualize_augmentations(img: Image, lbl: Image) -> None:
    figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 24))
    ax[0, 0].imshow(img[0])
    ax[0, 1].imshow(lbl[0], interpolation="nearest")
    ax[0, 0].set_title("Original image")
    ax[0, 1].set_title("Original mask")
    ax[0, 0].set_axis_off()
    ax[0, 1].set_axis_off()
    
    ax[1, 0].imshow(img[1])
    ax[1, 1].imshow(lbl[1], interpolation="nearest")
    ax[1, 0].set_title("Augmented image")
    ax[1, 1].set_title("Augmented mask")
    ax[1, 0].set_axis_off()
    ax[1, 1].set_axis_off()
        
    plt.tight_layout()
    plt.show()

def test1():
    # augments = A.Compose([
    #     A.RandomSizedCrop(min_max_height=(128, 256), height=256, width=512, p=0.5),
    #     A.HorizontalFlip(p=1),
    #     A.ShiftScaleRotate(p=1),
    #     A.RandomBrightnessContrast(p=1),
    # ])
    
    augments = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    ])
        
    img1 = Image.open('/home/renato/desenv/semanticSegmentation/datasets/pod500/train/run_1_47.jpg')
    lbl1 = Image.open('/home/renato/desenv/semanticSegmentation/datasets/pod500/train/run_1_47.png')
    
    augmented = augments(image=np.array(img1), mask=np.array(lbl1))
    return img1, lbl1, augmented
        
def test2():
    img1 = Image.open('/home/renato/desenv/semanticSegmentation/datasets/pod500/train/run_1_47.jpg')
    lbl1 = Image.open('/home/renato/desenv/semanticSegmentation/datasets/pod500/train/run_1_47.png')
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.566),
        A.RandomToneCurve(p=0.566),
        A.RandomBrightnessContrast(p=0.566),
    ])
    augmented = transform(image=np.array(img1), mask=np.array(lbl1))
    return img1, lbl1, augmented
    
if __name__ == '__main__':

    img1, lbl1, augmented = test2()
    print(np.array(img1).shape, augmented['image'].shape)
    print(type(augmented['image']), augmented['image'].dtype)
    visualize_augmentations([img1, augmented['image']], [lbl1, augmented['mask']])