import albumentations as A
import numpy as np
import random

from typing import Tuple, List

def get_augment_list() -> List:
    return [
            (A.Affine(), {'p': (0.0, 1.0)}),
            (A.CoarseDropout(), {'p': (0.0, 1.0)}),
            (A.ElasticTransform(), {'p': (0.0, 1.0)}),
            (A.Flip(), {'p': (0.0, 1.0)}),
            (A.GridDropout(), {'p': (0.0, 1.0)}),       
            (A.HorizontalFlip(), {'p': (0.0, 1.0)}), 
            (A.PiecewiseAffine(), {'p': (0.0, 1.0)}), 
            (A.Perspective(), {'p': (0.0, 1.0)}), 
            (A.RandomGridShuffle(), {'p': (0.0, 1.0)}),      
            (A.SafeRotate(), {'p': (0.0, 1.0)}),      
            (A.VerticalFlip(), {'p': (0.0, 1.0)}),                                              
            (A.OpticalDistortion(), {'distort_limit': (0.05, 0.20), 'p': (0.0, 1.0)}),                         
            (A.GridDistortion(), {'distort_limit': (0.3, 0.9), 'p': (0.0, 1.0)}),
            (A.ShiftScaleRotate(), {'shift_limit': (0, 1.0), 'scale_limit': (0, 1.0), 'rotate_limit': (0, 1.0), 'p': (0.0, 1.0)}),
            (A.RGBShift(), {'r_shift_limit': (0, 255), 'g_shift_limit': (0, 255), 'b_shift_limit': (0, 255), 'p': (0.0, 1.0)}),
            (A.RandomBrightnessContrast(), {'brightness_limit': (0, 1.0), 'contrast_limit': (0, 1.0), 'p': (0.0, 1.0)}),
            (A.Blur(), {'p': (0.0, 1.0)}),
            (A.CLAHE(), {'p': (0.0, 1.0)}),
            (A.ChannelDropout(), {'p': (0.0, 1.0)}),
            (A.ChannelShuffle(), {'p': (0.0, 1.0)}),
            (A.ColorJitter(), {'p': (0.0, 1.0)}),
            (A.Downscale(), {'p': (0.0, 1.0)}),
            (A.Emboss(), {'p': (0.0, 1.0)}),
            (A.Equalize(), {'p': (0.0, 1.0)}),
            (A.FancyPCA(), {'p': (0.0, 1.0)}),
            (A.GaussNoise(), {'p': (0.0, 1.0)}),
            (A.GaussianBlur(), {'p': (0.0, 1.0)}),
            (A.GlassBlur(), {'p': (0.0, 1.0)}),
            (A.HueSaturationValue(), {'p': (0.0, 1.0)}),
            (A.ISONoise(), {'p': (0.0, 1.0)}),
            (A.ImageCompression(), {'quality_lower': (90, 95), 'quality_upper': (95, 100), 'p': (0.0, 1.0)}),
            (A.InvertImg(), {'p': (0.0, 1.0)}),
            (A.MedianBlur(), {'p': (0.0, 1.0)}),
            (A.MotionBlur(), {'p': (0.0, 1.0)}),
            (A.MultiplicativeNoise(), {'p': (0.0, 1.0)}),
            (A.Posterize(), {'p': (0.0, 1.0)}),
            (A.RandomGamma(), {'p': (0.0, 1.0)}),
            (A.RandomRain(), {'p': (0.0, 1.0)}),
            (A.RandomSnow(), {'p': (0.0, 1.0)}),
            (A.RandomSunFlare(), {'p': (0.0, 1.0)}),
            (A.RandomToneCurve(), {'p': (0.0, 1.0)}),
            (A.Sharpen(), {'p': (0.0, 1.0)}),
            (A.Solarize(), {'p': (0.0, 1.0)}),
            (A.Superpixels(), {'p': (0.0, 1.0)}),
            (A.ToGray(), {'p': (0.0, 1.0)}),
            (A.ToSepia(), {'p': (0.0, 1.0)}),                
        ]    
  
def get_augment_list_where_p_eq_1() -> List:
    return [
            (A.Affine(p=1.0), {}),
            (A.CoarseDropout(p=1.0), {}),
            (A.ElasticTransform(p=1.0), {}),
            (A.Flip(p=1.0), {}),
            (A.GridDropout(p=1.0), {}),       
            (A.HorizontalFlip(p=1.0), {}), 
            (A.PiecewiseAffine(p=1.0), {}), 
            (A.Perspective(p=1.0), {}), 
            (A.RandomGridShuffle(p=1.0), {}),      
            (A.SafeRotate(p=1.0), {}),      
            (A.VerticalFlip(p=1.0), {}),                                              
            (A.OpticalDistortion(p=1.0), {'distort_limit': (0.05, 0.20)}),                         
            (A.GridDistortion(p=1.0), {'distort_limit': (0.3, 0.9)}),
            (A.ShiftScaleRotate(p=1.0), {'shift_limit': (0, 1.0), 'scale_limit': (0, 1.0), 'rotate_limit': (0, 1.0)}),
            (A.RGBShift(p=1.0), {'r_shift_limit': (0, 255), 'g_shift_limit': (0, 255), 'b_shift_limit': (0, 255)}),
            (A.RandomBrightnessContrast(p=1.0), {'brightness_limit': (0, 1.0), 'contrast_limit': (0, 1.0)}),
            (A.Blur(p=1.0), {}),
            (A.CLAHE(p=1.0), {}),
            (A.ChannelDropout(p=1.0), {}),
            (A.ChannelShuffle(p=1.0), {}),
            (A.ColorJitter(p=1.0), {}),
            (A.Downscale(p=1.0), {}),
            (A.Emboss(p=1.0), {}),
            (A.Equalize(p=1.0), {}),
            (A.FancyPCA(p=1.0), {}),
            (A.GaussNoise(p=1.0), {}),
            (A.GaussianBlur(p=1.0), {}),
            (A.GlassBlur(p=1.0), {}),
            (A.HueSaturationValue(p=1.0), {}),
            (A.ISONoise(p=1.0), {}),
            (A.ImageCompression(p=1.0), {'quality_lower': (90, 95), 'quality_upper': (95, 100), 'p': (0.0, 1.0)}),
            (A.InvertImg(p=1.0), {}),
            (A.MedianBlur(p=1.0), {}),
            (A.MotionBlur(p=1.0), {}),
            (A.MultiplicativeNoise(p=1.0), {}),
            (A.Posterize(p=1.0), {}),
            (A.RandomGamma(p=1.0), {}),
            (A.RandomRain(p=1.0), {}),
            (A.RandomSnow(p=1.0), {}),
            (A.RandomSunFlare(p=1.0), {}),
            (A.RandomToneCurve(p=1.0), {}),
            (A.Sharpen(p=1.0), {}),
            (A.Solarize(p=1.0), {}),
            (A.Superpixels(p=1.0), {}),
            (A.ToGray(p=1.0), {}),
            (A.ToSepia(p=1.0), {}),                
        ]
     
def get_augment_list_where_p_eq_half() -> List:
    print('get_augment_list_where_p_eq_half!:)')
    return [
            (A.Affine(p=0.5), {}),
            (A.CoarseDropout(p=0.5), {}),
            (A.ElasticTransform(p=0.5), {}),
            (A.Flip(p=0.5), {}),
            (A.GridDropout(p=0.5), {}),       
            (A.HorizontalFlip(p=0.5), {}), 
            (A.PiecewiseAffine(p=0.5), {}), 
            (A.Perspective(p=0.5), {}), 
            (A.RandomGridShuffle(p=0.5), {}),      
            (A.SafeRotate(p=0.5), {}),      
            (A.VerticalFlip(p=0.5), {}),                                              
            (A.OpticalDistortion(p=0.5), {'distort_limit': (0.05, 0.20)}),                         
            (A.GridDistortion(p=0.5), {'distort_limit': (0.3, 0.9)}),
            (A.ShiftScaleRotate(p=0.5), {'shift_limit': (0, 1.0), 'scale_limit': (0, 1.0), 'rotate_limit': (0, 1.0)}),
            (A.RGBShift(p=0.5), {'r_shift_limit': (0, 255), 'g_shift_limit': (0, 255), 'b_shift_limit': (0, 255)}),
            (A.RandomBrightnessContrast(p=0.5), {'brightness_limit': (0, 1.0), 'contrast_limit': (0, 1.0)}),
            (A.Blur(p=0.5), {}),
            (A.CLAHE(p=0.5), {}),
            (A.ChannelDropout(p=0.5), {}),
            (A.ChannelShuffle(p=0.5), {}),
            (A.ColorJitter(p=0.5), {}),
            (A.Downscale(p=0.5), {}),
            (A.Emboss(p=0.5), {}),
            (A.Equalize(p=0.5), {}),
            (A.FancyPCA(p=0.5), {}),
            (A.GaussNoise(p=0.5), {}),
            (A.GaussianBlur(p=0.5), {}),
            (A.GlassBlur(p=0.5), {}),
            (A.HueSaturationValue(p=0.5), {}),
            (A.ISONoise(p=0.5), {}),
            (A.ImageCompression(p=0.5), {'quality_lower': (90, 95), 'quality_upper': (95, 100), 'p': (0.0, 1.0)}),
            (A.InvertImg(p=0.5), {}),
            (A.MedianBlur(p=0.5), {}),
            (A.MotionBlur(p=0.5), {}),
            (A.MultiplicativeNoise(p=0.5), {}),
            (A.Posterize(p=0.5), {}),
            (A.RandomGamma(p=0.5), {}),
            (A.RandomRain(p=0.5), {}),
            (A.RandomSnow(p=0.5), {}),
            (A.RandomSunFlare(p=0.5), {}),
            (A.RandomToneCurve(p=0.5), {}),
            (A.Sharpen(p=0.5), {}),
            (A.Solarize(p=0.5), {}),
            (A.Superpixels(p=0.5), {}),
            (A.ToGray(p=0.5), {}),
            (A.ToSepia(p=0.5), {}),                
        ]

class RandAugment:
    def __init__(self, imgsz = Tuple[int, int], n: int = 3, m: int = 5, augment_list=get_augment_list_where_p_eq_half()):
        self.n = n
        self.m = m      # [0, 30]
        self.imgsz = imgsz
        self.augment_list = augment_list
         
    def __call__(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ops = random.sample(self.augment_list, k=self.n)
        for op, params in ops:
            kargs = {}
            for pname in params.keys():
                minval, maxval = params[pname]
                val = (float(self.m) / 30) * float(maxval - minval) + minval
                kargs[pname] = val
            # print(op.__class__.__name__, img.shape, img.dtype, lbl.shape, lbl.dtype, kargs)            
            augmented = op(image=img, mask=lbl, **kargs)
            img = augmented['image']
            lbl = augmented['mask']

        return img, lbl    