import imgaug as ia
ia.seed(9)
import imgaug.augmenters as iaa
import imgaug.parameters as iap 


def get_trans():
    dict_trans = {
        'art_aug': artistic(),
        'ari_dr_aug': arithmetic_drop(),
        'ari_ns_aug': arithmetic_noise(),
        'ari_wh_aug': arithmetic_whole(),
        'bld_aug': blend(),
        'blu_aug': blur(),
        'clr_aug': color(),
        'geo_aug': geometric(),
        'meta_aug': meta_channel_shuffle(), 
        'wea_aug': weather(),
    }
    return dict_trans


## ---------------
def arithmetic_whole():
    return [
        iaa.Add((-40, 40), per_channel=0.5),
        iaa.AddElementwise((-40, 40), per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=(0.05, 0.2*255)),
        iaa.AdditiveLaplaceNoise(scale=(0.05, 0.2*255)),
        iaa.AdditivePoissonNoise(lam=(10, 40), per_channel=True),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
        iaa.Invert(0.25, per_channel=0.5),
        iaa.Solarize(0.5, threshold=(32, 128)),
        iaa.JpegCompression(compression=(70, 90))
    ]

def arithmetic_drop():
    return [
        iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False),
        iaa.Cutout(fill_mode="constant", cval=(0, 255),fill_per_channel=0.5),
        iaa.Cutout(fill_mode="gaussian", fill_per_channel=True),
        iaa.Dropout(p=(0, 0.2), per_channel=0.5),
        iaa.CoarseDropout((0.01, 0.05), size_percent=(0.02, 0.03), per_channel=0.5),
        # iaa.Dropout2d(p=0.5),
    ]

def arithmetic_noise():
    return [
        iaa.ReplaceElementwise(0.1, [0, 255], per_channel=0.5),
        iaa.ReplaceElementwise(
            iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
            iap.Normal(128, 0.4*128), per_channel=0.5),
        iaa.ImpulseNoise(0.1),
        iaa.SaltAndPepper(0.1, per_channel=True),
        iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
        iaa.Salt(0.1),
        iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1), per_channel=True),
        iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1)),
        iaa.Pepper(0.1),
        iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1)),
    ]

## ---------------
def artistic():
    return [iaa.Cartoon()]

## ---------------
def blend():
    return [
        iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0)),
        iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(100), background=iaa.Multiply(0.2)),
        iaa.BlendAlphaMask(iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),iaa.Clouds()),
        iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(100)),
        iaa.BlendAlphaSimplexNoise(foreground=iaa.EdgeDetect(1.0), sigmoid_thresh=iap.Normal(10.0, 5.0)),
        iaa.BlendAlphaFrequencyNoise(foreground=iaa.EdgeDetect(1.0), sigmoid_thresh=iap.Normal(10.0, 5.0)),
        iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0)),
        iaa.BlendAlphaSomeColors(iaa.MultiplySaturation(0.5), iaa.MultiplySaturation(1.5)),
        iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-100, 100))),
        iaa.BlendAlphaVerticalLinearGradient(iaa.Clouds(), start_at=(0.15, 0.35), end_at=0.0),
        iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),foreground=iaa.AddToHue((-100, 100))),
    ]

def blur():
    return [
        iaa.GaussianBlur(sigma=(0.01, 2.5)),
        iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
    ]

def weather():
    return [
        iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03)),
        iaa.Rain(speed=(0.1, 0.3)),
        iaa.FastSnowyLandscape(lightness_threshold=140, lightness_multiplier=2.5),
        iaa.Clouds(),
        iaa.Fog()
    ]

def color():
    return [
        iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", 
                           children=iaa.WithChannels(0,iaa.Add((0, 50)))),
        iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.WithHueAndSaturation([
            iaa.WithChannels(0, iaa.Add((-30, 10))),
            iaa.WithChannels(1, [
                iaa.Multiply((0.5, 1.5)),
                iaa.LinearContrast((0.75, 1.25))
                ])
            ]),
        iaa.RemoveSaturation(1.0),
        iaa.AddToSaturation((-50, 50)),
        # iaa.ChangeColorTemperature((1100, 10000)),# bug
        iaa.KMeansColorQuantization(n_colors=(4, 16))
    ]


def geometric():
    return [
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.Affine(scale=(0.7, 1.2)),
        iaa.Affine(rotate=(-20, 20)),
        iaa.Affine(shear=(-16, 16)),
        # iaa.PiecewiseAffine(scale=(0.01, 0.05)),# bug
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.Fliplr(),
        iaa.Flipud(),
        iaa.CropAndPad(percent=(0, 0.2), pad_mode=["constant", "edge"], pad_cval=(0, 128)),
        
    ]

def meta_channel_shuffle():
    return [iaa.ChannelShuffle(1, channels=[0, 1])]    
