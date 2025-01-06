import albumentations as alb

def get_trans():
    dict_trans = {
        'ari_dr_aug': arithmetic_drop(),#
        'ari_ns_aug': arithmetic_noise(),#
        'ari_wh_aug': arithmetic_crop(),#
        'blu_aug': blur(),#
        'clr_aug': color(),
        'geo_aug': geometric(),#
        'wea_aug': weather(),
    }
    return dict_trans

def color():
    return [
        alb.ColorJitter(p=1.0, brightness=(0.8, 1.2), contrast=0, saturation=0, hue=0), # brightness=(0.8,1.2)
        alb.ColorJitter(p=1.0, brightness=0, contrast=(0.8, 1.2), saturation=0, hue=0), # contrast
        alb.ColorJitter(p=1.0, brightness=0, contrast=0, saturation=(0.8, 1.2), hue=0), # saturation
        alb.ColorJitter(p=1.0, brightness=0, contrast=0, saturation=0, hue=(-0.5, 0.5)), # hue
        alb.HueSaturationValue(p=1.),
        alb.RandomBrightnessContrast(p=1.)
    ]


def blur():
    return [
        alb.Blur(p=1.0),
        alb.GaussianBlur(p=1.0),
        alb.GlassBlur(p=1.0)
    ]


def geometric():
    return [
        alb.Affine(scale=(0.1,0.4), keep_ratio=True, translate_percent=None, rotate=0, shear=0, p=1.0), # scale
        alb.Affine(scale=1, translate_percent=(0.1, 0.5), rotate=0, shear=0, p=1.0), # translate
        alb.Affine(scale=1, translate_percent=None, rotate=(-30, 30), shear=0, p=1.0), # rotate
        alb.Affine(scale=1, translate_percent=None, rotate=0, shear=(-30, 30), p=1.0) # shear
    ]

def arithmetic_crop():
    return [
        alb.CropAndPad(p=1.0, percent=0.2)
    ]

def arithmetic_drop():
    return [
        alb.CoarseDropout(p=1., num_holes_range=(1,3)),
        alb.Erasing(p=1.0)
    ]

def arithmetic_noise():
    return [
        alb.AdditiveNoise(p=1.0, noise_type='uniform'),
        alb.AdditiveNoise(p=1.0, noise_type='gaussian'),
        alb.AdditiveNoise(p=1.0, noise_type='laplace'),
        alb.AdditiveNoise(p=1.0, noise_type='beta'),
        alb.GaussNoise(p=1.0),
        alb.ISONoise(p=1.0),
        alb.SaltAndPepper(p=1.0)
    ]


def weather():
    return [
        alb.RandomFog(p=1.0),
        alb.RandomRain(p=1.0),
        alb.RandomSnow(p=1.0),
        alb.RandomSunFlare(p=1.0),
    ]