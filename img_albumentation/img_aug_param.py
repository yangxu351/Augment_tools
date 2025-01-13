import albumentations as alb
import cv2
import random 


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
        'scale': scale,#
        'translate': translate,#
        'crop': crop,
        'rotate': rotate,#
        'flip': hv_flip,#
        'brightness': brightness,#
        'contrast': contrast,
        'saturation': saturation,
        'hue': hue,
        'gaussain_noise':gaussian_noise,
        'poisson_noise':poisson_noise,
        'saltpepper':saltpepper,
        'gaussain_blur':gaussian_blur,
        'mean_blur': median_blur,
        'fog':random_fog,
        'rain':random_rain,
        'snow':random_snow,
        
        'shear':shear,
        'clahe':clahe,
        'defocus':defocus,
        'glassblur':glassblur,
        'multicative_noise':multicative_noise,
        'illumination':illumination,
        'shadow':plasma_shadow,
        'posterize':posterize,
        'sun':random_sun_flare,
        'beta_noise':beta_noise,
        'rain_spatter':rain_spatter,
        'mud_spatter':mud_spatter,
        'random_gravel':random_gravel,
        'super_pixels':super_pixels,
        'sepia':sepia,
        'deformation':deformation,
    }
    return dict_trans

######## geometric 
def scale(direction='down'): # 缩放 1 means no change
    if direction=='up':
        scale=(1.1, 1.3)
    else: # 'down'
        scale=(0.7, 0.9)
    return alb.Affine(scale=scale, keep_ratio=True, border_mode=cv2.BORDER_REPLICATE, p=1.)


def translate(direction='right'): # 平移
    if direction=='left':
        translate_px=(-50,-10)
    else:#'right'
        translate_px=(10,50)
    return alb.Affine(translate_px=translate_px, border_mode=cv2.BORDER_REPLICATE, p=1.)


def crop(degree='low'): # 剪切 
    if degree=='high':
        px = ([0.2,0.4],[0.2,0.4],[0.2,0.4],[0.2,0.4])
    else: # low
        px = ([0.05,0.15],[0.05,0.15],[0.05,0.15],[0.05,0.15])
    return alb.CropAndPad(px=px, boder_mode=cv2.BORDER_REPLICATE, p=1)


def rotate(direction='right'): # 旋转
    if direction=='left':
        limit=(10,90)
    else:#'right'
        limit=(-90,-10)
    return alb.Rotate(limit=limit, p=1.)


def hv_flip(direction="V"): # 翻转
    '''
    hori_or_vert ["V", "H"]
    '''
    if direction == "H":
        return alb.HorizontalFlip(p=1.)
    else: # "V"
        return alb.VerticalFlip(p=1.)


######## color
def brightness(degree='low'):
    if degree=='high':
        brightness=(1.1, 1.3)
    else:# 'low'
        brightness=(0.7, 0.9)
    return alb.ColorJitter(brightness=brightness, saturation=0, contrast=0, hue=0, p=1)
        

def contrast(degree='low'):
    if degree=='high':
        contrast=(1.1, 1.3)
    else:# 'low'
        contrast=(0.7, 0.9)
    return alb.ColorJitter(brightness=0, saturation=0, contrast=contrast, hue=0, p=1)


def saturation(degree='low'):
    if degree=='high':
        saturation=(1.1, 1.3)
    else:# 'low'
        saturation=(0.7, 0.9)
    return alb.ColorJitter(brightness=0, saturation=saturation, contrast=0, hue=0, p=1)


def hue(degree='low'):
    if degree=='high':
        hue=(0.1, 0.5)
    else:# 'low'
        hue=(-0.5, -0.1)
    return alb.ColorJitter(brightness=0, saturation=0, contrast=0, hue=hue, p=1)



######## noise 高斯噪声、椒盐噪声、泊松噪声、高斯模糊、均值模糊、
def gaussian_noise(degree='low'):
    if degree=='high':
        std_range=(0.05, 0.15)
    else:
        std_range=(0.2, 0.3)
    return alb.GaussNoise(std_range=std_range, mean_range=0, p=1)


def poisson_noise(degree='low'):
    if degree=='high':
        scale_range=(0.4,0.5)
    else:# low
        scale_range=(0.2,0.3)
    return alb.ShotNoise(scale_range=scale_range, p=1.)


def saltpepper(degree='low'):
    if degree=='high':
        amount=(0.04,0.06)
        salt_vs_pepper=(0.4,0.6)
    else: # low
        amount=(0.01,0.03) 
        salt_vs_pepper=(0.4,0.6)
    return alb.SaltAndPepper(amount=amount, salt_vs_pepper=salt_vs_pepper, p=1)


def gaussian_blur(degree='low'):
    if degree=='high':
        blur_limit=(5,7)
        sigma_limit=(0.1,2)
    else: # low
        blur_limit=(3,4)
        sigma_limit=(0.1,2)
    return alb.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma_limit, p=1)


def median_blur(degree='low'):
    if degree=='high':
        blur_limit=(5,7)
    else: # low
        blur_limit=(3,4)
    return alb.median_blur(blur_limit=blur_limit, p=1.)



######### weather
def random_fog(degree='low'):
    if degree=='high':
        fog_coef_range=(0.5,0.8)
        alpha_coef=0.58
    else: # low
        fog_coef_range=(0.2,0.4)
        alpha_coef=0.28
    return alb.RandomFog(fog_coef_range=fog_coef_range, alpha_coef=alpha_coef, p=1.)
    

def random_rain(degree='low'):
    '''
    brightness_coefficient (0.1,1)
    rain_type=Literal["drizzle", "heavy", "torrential", "default"]
    '''
    if degree=='high':
        blur_value=random.choice([5,6])
        brightness_coefficient=0.6
        rain_type='heavy'
    else: # low
        blur_value=random.choice([2,3])
        brightness_coefficient=0.8
        rain_type='drizzle'
    return alb.RandomRain(blur_value=blur_value, brightness_coefficient=brightness_coefficient, rain_type=rain_type, p=1)


def random_snow(degree='low'):
    '''
    snow_point_range (0.1,1)
    method []'bleach','texture']
    '''
    if degree=='high':
        snow_point_range=(0.2,0.3) 
        brightness_coeff=2 
    else:
        snow_point_range=(0.1,0.15) 
        brightness_coeff=1 
    return alb.RandomSnow(snow_point_range=snow_point_range, brightness_coeff=brightness_coeff, method='texture', p=1)


# FIXME:非要求的
#### additive
def shear(direction='left'): # 扭曲，剪切 
    if direction == 'right':
        shear=(-45, -5)
    else:# 'left'
        shear=(5, 45)
    return alb.Affine(shear=shear, border_mode=cv2.BORDER_REPLICATE, p=1.)


def clahe(degree='low'): 
    '''
    直方图均衡
    '''
    if degree=='high':
        clip_limit=(1,3)
    else:# 'low'
        clip_limit=(4,6)
    return alb.CLAHE(clip_limit=clip_limit, tile_grid_size=(15,15), p=1.)


def defocus(degree='low'):
    if degree=='high':
        radius=(3,5)
    else:# 'low'
        radius=(6,10)
    return alb.Defocus(radius=radius, p=1.)


def glassblur(degree='low'):
    if degree=='high':
        sigma = 0.1
        max_delta = 2
    else:# 'low'
        sigma = 0.7
        max_delta = 4
    return alb.GlassBlur(sigma=sigma, max_delta=max_delta, p=1.)


def multicative_noise(degree='low'):
    if degree=='high':
        multiplier=(1.1, 1.3)
    else:# 'low'
        multiplier=(0.7, 0.9)
    return alb.MultiplicativeNoise(multiplier=multiplier, per_channel=True, p=1.0)


def illumination(degree='low'):
    if degree=='high':
        mode='blackbody'
    else:# 'low'
        mode='cied'
    return alb.PlanckianJitter(mode=mode, p=1.)    


def plasma_shadow(degree='low'):
    if degree=='high':
        shadow_intensity_range=(0.5,0.7)
        plasma_size= 128
    else:# 'low'
        shadow_intensity_range=(0.3,0.45)
        plasma_size= 128
    return alb.PlasmaShadow(shadow_intensity_range=shadow_intensity_range, plasma_size=plasma_size, p=1.)


def posterize(degree='low'):
    if degree=='high':
        num_bits=(3,4)
    else:# 'low'
        num_bits=(4,5)
    return alb.Posterize(num_bits=num_bits, p=1.)


def random_sun_flare(degree='low'):
    if degree=='high':
        src_radius=400
        num_flare_circles_range=(3,4)
    else:# 'low'
        src_radius=200
        num_flare_circles_range=(1,2)
    return alb.RandomSunFlare(src_radius=src_radius, num_flare_circles_range=num_flare_circles_range, method='physics_based', p=1.)

def beta_noise(degree='low'):
    if degree=='high':
        noise_params = {
            'alpha_range':(0.5, 1.5),
            'beta_range': (0.5, 1.5),
            'scale_range': (0.3, 0.4)
        }
    else:# low
        noise_params = {
            'alpha_range':(0.5, 1.5),
            'beta_range': (0.5, 1.5),
            'scale_range': (0.15, 0.25)
        }
    return alb.AdditiveNoise(noise_type='beta', spatial_mode='shared', noise_params=noise_params, p=1.)


def rain_spatter(degree='low'):
    '''
    它模拟了以雨的形式遮挡镜头的情况
    '''
    if degree=='high':
        gauss_sigma=(2,5)
    else:# 'low'
        gauss_sigma=(6,8)
    return alb.Spatter(gauss_sigma=gauss_sigma, mode='rain', p=1.)

def mud_spatter(degree='low'):
    '''
    它模拟了以泥的形式遮挡镜头的情况
    '''
    if degree=='high':
        gauss_sigma=(1,2)
    else:# 'low'
        gauss_sigma=(3,4)
    return alb.Spatter(gauss_sigma=gauss_sigma, mode='mud', p=1.)


def random_gravel(degree='low'):
    if degree=='high':
        num_patches=random.choice([3,4,5])
    else:# 'low'
        num_patches=random.choice([1,2])
    return alb.RandomGravel(number_of_patches=num_patches, p=1.)


def super_pixels(degree='low'):
    if degree=='high':
        p_replace=(0.08,0.13)
    else:# 'low'
        p_replace=(0.01,0.07)
    return alb.Superpixels(p_replace=p_replace, p=1.)


def sepia(para=None):
    '''
    棕褐色滤镜
    '''
    return alb.ToSepia(p=1.)


def deformation(degree='low'):
    if degree=='high':
        scale_range=(0.3,0.4)
    else:# 'low'
        scale_range=(0.15,0.25)
    return alb.ThinPlateSpline(scale_range=scale_range, p=1.)
