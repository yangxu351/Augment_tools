import os
from PIL import Image
import numpy as np
from aug_preprocess.utils import make_dir_if_not_exist
from img_albumentation import img_aug_param as albaug
import albumentations as alb
import datetime

def alb_aug_cls(base_dir, split='val', method_params={}, combine=False, exec_num=1, dst_img_suffix='.jpg'):
    '''
    method_params dict {
                        'rotate': {'limit':(-90,90)}
                        'crop': {'left_px':-10, 'right_px':10, 'top_px':-10, 'bottom_px':10}
                        }
    '''

    source_dir = os.path.join(base_dir, split)
    save_dir = os.path.join(base_dir , split + '_albaug_param')
    make_dir_if_not_exist(save_dir, rm=False)

    all_trans = albaug.get_trans() 
    dict_trans = {}
    for method, params in method_params.items():
        dict_trans[method] = all_trans[method](**params)

    cat_folders = os.listdir(source_dir)
    
    for cf in cat_folders:
        img_names = os.listdir(os.path.join(source_dir, cf))
        img_names.sort()
        img_list = []
        for n in img_names:# FIXME:
            img = np.array(Image.open(os.path.join(source_dir,cf,n)))
            img_list.append(img)

        for en in range(exec_num):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%m%d-%H%M")
            # if combine:
            #     aug = alb.Compose([*dict_trans.values()])
            #     new_img_list = aug(images=img_list)['images']
            #     for nx, n_img in enumerate(new_img_list):
            #         n_file = os.path.join(c_save_dir, f"{img_names[nx].split('.')[0]}_ex{en}_comb_{formatted_time}{dst_img_suffix}")
            #         Image.fromarray(n_img).save(n_file) 
            # else: # independently
            for key, aug in dict_trans.items():
                c_save_dir = os.path.join(save_dir, cf, key)
                make_dir_if_not_exist(c_save_dir, rm=True)
                new_img_list = aug(images=img_list)['images']
                for nx, n_img in enumerate(new_img_list):
                    n_file = os.path.join(c_save_dir, f"{img_names[nx].split('.')[0]}_ex{en+1}_{formatted_time}{dst_img_suffix}")
                    Image.fromarray(n_img).save(n_file)        

        
