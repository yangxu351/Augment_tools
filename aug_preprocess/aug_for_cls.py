import os
from PIL import Image
import numpy as np
from img_imgaug.imgcls_aug import get_trans
from utils import make_dir_if_not_exist


def aug_cls(base_dir, split='val', dst_img_suffix='.jpg'):
    source_dir = os.path.join(base_dir, split)
    save_dir = os.path.join(base_dir + '_aug', split)
    make_dir_if_not_exist(save_dir, rm=False)

    dict_trans = get_trans()
    cat_folders = os.listdir(source_dir)
    for cf in cat_folders:
        ori_imgs = os.listdir(os.path.join(source_dir, cf))
        ori_imgs.sort()
        img_names = [os.path.basename(om).split('.')[0] for om in ori_imgs]
        img_list = []
        for f in ori_imgs:
            img = np.array(Image.open(f))
            img_list.append(img)

        c_save_dir = os.path.join(save_dir, cf)
        make_dir_if_not_exist(c_save_dir, rm=True)
        for key, trans in dict_trans.items():
            for ax, aug in enumerate(trans):
                new_img_list = aug(images=img_list)
                for nx, n_img in enumerate(new_img_list):
                    n_file = os.path.join(c_save_dir, f'{img_names[nx]}_{key}_{ax}{dst_img_suffix}')
                    Image.fromarray(n_img).save(n_file)        

    
    
