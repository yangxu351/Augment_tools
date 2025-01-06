import os
from PIL import Image
import numpy as np
from aug_preprocess.utils import make_dir_if_not_exist


def imgaug_cls(base_dir, split='val', dst_img_suffix='.jpg'):
    from img_imgaug import imgcls_aug as imgaug

    source_dir = os.path.join(base_dir, split)
    save_dir = os.path.join(base_dir, split + '_imgaug')
    make_dir_if_not_exist(save_dir, rm=False)

    dict_trans = imgaug.get_trans()
    cat_folders = os.listdir(source_dir)
    for cf in cat_folders:
        ori_imgs = os.listdir(os.path.join(source_dir, cf))
        ori_imgs.sort()
        img_names = [os.path.basename(om).split('.')[0] for om in ori_imgs]
        img_list = []
        for n in ori_imgs:
            img = np.array(Image.open(os.path.join(source_dir,cf, n)))
            img_list.append(img)

        c_save_dir = os.path.join(save_dir, cf)
        make_dir_if_not_exist(c_save_dir, rm=True)
        for key, trans in dict_trans.items():
            for ax, aug in enumerate(trans):
                new_img_list = aug(images=img_list)
                for nx, n_img in enumerate(new_img_list):
                    n_file = os.path.join(c_save_dir, f'{img_names[nx]}_{key}_{ax}{dst_img_suffix}')
                    Image.fromarray(n_img).save(n_file)        

    
def alb_aug_cls(base_dir, split='val', dst_img_suffix='.jpg'):
    from img_albumentation import imgcls_aug as albaug

    source_dir = os.path.join(base_dir, split)
    save_dir = os.path.join(base_dir , split + '_albaug')
    make_dir_if_not_exist(save_dir, rm=False)

    dict_trans = albaug.get_trans()
    cat_folders = os.listdir(source_dir)
    for cf in cat_folders:
        img_names = os.listdir(os.path.join(source_dir, cf))
        img_names.sort()
        img_list = []
        for n in img_names[:2]:
            img = np.array(Image.open(os.path.join(source_dir,cf,n)))
            img_list.append(img)

        c_save_dir = os.path.join(save_dir, cf)
        make_dir_if_not_exist(c_save_dir, rm=True)
        for key, trans in dict_trans.items():
            for ax, aug in enumerate(trans):
                new_img_list = aug(images=img_list)['images']
                for nx, n_img in enumerate(new_img_list):
                    n_file = os.path.join(c_save_dir, f"{img_names[nx].split('.')[0]}_{key}_{ax}{dst_img_suffix}")
                    Image.fromarray(n_img).save(n_file)        

        
