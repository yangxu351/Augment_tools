import Augmentor as aug 


def get_trans():
    dict_trans = {
        'rot_aug': rotating(),#
        'skew_aug': shearing(),#
        'crop_aug': cropping(),#
        'mir_aug': mirroring(),#
        'dist_aug': distortions(),
    }
    return dict_trans

def augment(data_path, num_samples):
    pipe = aug.Pipeline(data_path)
    

def rotating(pipe):
    return [
        pipe.rotate(p=1.0)
    ]

def skewing(data_path):
    pipe = aug.Pipeline(data_path)
    return [
        pipe.skew_tilt(p=1.0)
    ]

def distortions(pipe):
    return pipe.random_distortion(p=1.)

def shearing(pipe):
    return pipe.shear(p=1.0)

def cropping(pipe):
    return [
        pipe.crop_centre(p=1.0),
        pipe.crop_random(p=1.0),
        pipe.crop_by_size(p=1.0)
    ]

def mirroring(pipe):
    return [
        pipe.flip_left_right(p=1.0),
        pipe.flip_top_bottom(p=1.0),
        pipe.flip_random(p=1.0)
    ]