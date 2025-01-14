import os
from PIL import Image
import numpy as np 
from aug_preprocess.utils import make_dir_if_not_exist
from img_albumentation import img_aug_param as albaug
import datetime

    
def alb_aug_msk(base_dir, split='val', method_params={}, combine=False, exec_num=1):
    src_img_dir = os.path.join(base_dir, split, 'images')
    src_msk_dir = os.path.join(base_dir, split, 'masks')
    save_img_dir = os.path.join(base_dir, split + '_albaug_param', 'images')
    # make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = os.path.join(base_dir,split  + '_albaug_param', 'masks')
    # make_dir_if_not_exist(save_lbl_dir, rm=True)

    img_names = os.listdir(src_img_dir)
    img_names.sort()
    msk_names = os.listdir(src_msk_dir)
    msk_names.sort()
    assert len(img_names)==len(msk_names), 'the quantity of images is different from that of masks'
    
    all_trans = albaug.get_trans()
    dict_trans = {}
    for method, params in method_params.items():
        dict_trans[method] = all_trans[method](**params)
    # trans_indexes= ['ari_dr_aug','ari_ns_aug','ari_wh_aug','blu_aug','clr_aug','wea_aug']
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m%d-%H%M")

    for key, aug in dict_trans.items():
        s_save_img_dir = os.path.join(save_img_dir, key)
        make_dir_if_not_exist(s_save_img_dir, rm=True)
        s_save_lbl_dir = os.path.join(save_lbl_dir, key)
        make_dir_if_not_exist(s_save_lbl_dir, rm=True)
        for ix in range(len(img_names)):# len(img_names)
            img = np.array(Image.open(os.path.join(src_img_dir, img_names[ix])))
            img_name, img_suffix = img_names[ix].split('.')
            # print(' ori shape',  img.shape) # h,w,c
            # img_list.append(img)
            
            msk_file = os.path.join(src_msk_dir, f'{msk_names[ix]}')
            msk_name, msk_suffix = msk_names[ix].split('.')
            # print('--msk---', msk_file)
            msk = np.array(Image.open(msk_file))
            # map_list.append(msk)
        
            for en in range(exec_num):
                dict_img_msk = aug(image=img, mask=msk)
                new_img = dict_img_msk["image"]
                new_msk = dict_img_msk["mask"]
                n_img_file = os.path.join(s_save_img_dir, f'{img_name}_ex{en+1}_{formatted_time}.{img_suffix}')
                Image.fromarray(new_img).save(n_img_file)
                n_msk_file = os.path.join(s_save_lbl_dir, f'{msk_name}_ex{en+1}_{formatted_time}.{msk_suffix}')
                Image.fromarray(new_msk).save(n_msk_file)# mask 不变