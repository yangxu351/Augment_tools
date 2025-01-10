import albumentations as alb
import cv2


"""
图像旋转、翻转、缩放、平移、剪切、
亮度调整、对比度调整、饱和度调整、色相变换、
高斯噪声、椒盐噪声、泊松噪声、高斯模糊、均值模糊、
雨、雪、雾
"""

# def get_trans(combine=False, params={}, exec_num=1):
#     for k,v in params.items():
# A.Sequential([ for _ in range(exec_num)])


def get_trans():
    dict_trans = {
        'rotate': rotate,#
        'flip': hv_flip,#
        'scale': scale,#
        'translate': translate,#
        'crop': crop,
        'brightness': brightness,#
        'contrast': contrast,
        'saturation': saturation,
        'hue': hue,
        'gaussain_noise':gaussian_noise,
        'saltpepper':saltpepper,
        'poisson_noise':poisson_noise,
        'gaussain_blur':gaussian_blur,
        'mean_blur': median_blur,
        'rain':random_rain,
        'snow':random_snow,
        'fog':random_fog
    }
    return dict_trans

######## geometric 
def scale(scale=(0.5, 1.5)): # 缩放 1 means no change
    return alb.Affine(scale=scale, keep_ratio=True, border_mode=cv2.BORDER_REPLICATE, p=1.)

def translate(translate_px=(-50,50)): # 平移
    return alb.Affine(translate_px=translate_px, border_mode=cv2.BORDER_REPLICATE, p=1.)

def crop(left_px=-10, right_px=10, top_px=-10, bottom_px=10): # 剪切 
    return alb.CropAndPad(px=(left_px, right_px, top_px, bottom_px), boder_mode=cv2.BORDER_REPLICATE, p=1)

def rotate(limit=(-90,90)): # 旋转
    return alb.Rotate(limit=limit, p=1.)

def hv_flip(hori_or_vert="V"): # 翻转
    '''
    hori_or_vert ["V", "H"]
    '''
    if hori_or_vert == "H":
        return alb.HorizontalFlip(p=1.)
    else: # "V"
        return alb.VerticalFlip(p=1.)

def shear(shear=(-45, 45)): # 扭曲，剪切
    return alb.Affine(shear=shear, border_mode=cv2.BORDER_REPLICATE, p=1.)


######## color
def brightness(brightness=(0.8, 1.2)):
    return alb.ColorJitter(brightness=brightness, saturation=0, contrast=0, hue=0, p=1)

def contrast(contrast=(0.8, 1.2)):
    return alb.ColorJitter(brightness=0, saturation=0, contrast=contrast, hue=0, p=1)

def saturation(saturation=(0.8, 1.2)):
    return alb.ColorJitter(brightness=0, saturation=saturation, contrast=0, hue=0, p=1)

def hue(hue=(0.1, 0.5)):
    return alb.ColorJitter(brightness=0, saturation=0, contrast=0, hue=hue, p=1)

######## noise 高斯噪声、椒盐噪声、泊松噪声、高斯模糊、均值模糊、

def gaussian_noise(var_limit=None, mean=None, std_range=(0.2, 0.44), mean_range=0):
    return alb.GaussNoise(var_limit=var_limit, mean=mean, std_range=std_range, mean_range=mean_range, p=1)

def poisson_noise():
    return alb.AdditiveNoise(noise_type='beta', spatial_mode='shared', p=1.)

def saltpepper(amount=(0.01,0.06), salt_vs_pepper=(0.4,0.6)):
    return alb.SaltAndPepper(amount=amount, salt_vs_pepper=salt_vs_pepper, p=1)

def gaussian_blur(blur_limit=(3,7), sigma_limit=(0.1,2)):
    return alb.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma_limit, p=1)

def median_blur(blur_limit=(3,7)):
    return alb.median_blur(blur_limit=blur_limit, p=1.)

######### weather
def random_fog(fog_coef_range=(0,1), alpha_coef=0.08): # alpha_coef=(0,1)
    return alb.RandomFog(fog_coef_range=fog_coef_range, alpha_coef=alpha_coef, p=1.)
    
def random_rain(slant_range=(-10,10), drop_length=20, drop_width=1, blur_value=7, brightness_coefficient=0.7, rain_type='default'): 
    '''
    brightness_coefficient (0.1,1)
    rain_type=Literal["drizzle", "heavy", "torrential", "default"]
    '''
    return alb.RandomRain(slant_range=slant_range, drop_length=drop_length, drop_width=drop_width, blur_value=blur_value, brightness_coefficient=brightness_coefficient, rain_type=rain_type, p=1)

def random_snow(snow_point_range=(0.1,0.3), brightness_coeff=2.5, method='texture'):
    '''
    snow_point_range (0.1,1)
    method []'bleach','texture']
    '''
    return alb.RandomSnow(snow_point_range=snow_point_range, brightness_coeff=brightness_coeff, method=method, p=1)

