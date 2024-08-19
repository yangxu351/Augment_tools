import os
import shutil
import argparse
import ast
import glob
import cv2
from img_imgaug.imgaug import *
# from Img_albumentations import albumentations as AB


def make_dir_if_not_exist(mf, rm=False):
    if not os.path.exists(mf):
        os.makedirs(mf)
    else:
        if rm: # remove folder if exists
            shutil.rmtree(mf)
            os.mkdir(mf)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/MSTAR')
    parser.add_argument('--aug_list', type=ast.literal_eval, 
                        default="['art_cartoon', 'ari_drop', 'ari_noise', 'ari_while', 'blend', 'blur', 'color', 'geometric', 'weather', 'meta']", 
                        help='art')
    parser.add_argument('--img_suffix_list', type=ast.literal_eval, default="['.png', '.bmp', '.jpg', '.JPG', '.PNG']", help='.bmp .png .jpg')
    parser.add_argument('--src_img_suffix', type=str, default='.JPG', help='.png .jpg')
    parser.add_argument('--dst_suffix', type=str, default='.JPG', help='.png .jpg')
    args = parser.parse_args()
    
    source_dir = os.path.join(args.base_dir, 'test')
    save_dir = source_dir + '_aug'
    make_dir_if_not_exist(save_dir, rm=False)

    
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

    cat_folders = os.listdir(source_dir)
    for cf in cat_folders:
        ori_imgs = glob.glob(os.path.join(source_dir, cf, f'*{args.src_img_suffix}'))
        ori_imgs.sort()
        img_names = [os.path.basename(om).split('.')[0] for om in ori_imgs]
        img_list = []
        for f in ori_imgs:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)

        c_save_dir = os.path.join(save_dir, cf)
        make_dir_if_not_exist(c_save_dir, rm=False)
        for key, trans in dict_trans.items():
            for ax, aug in enumerate(trans):
                new_img_list = aug(images=img_list)
                for nx, n_img in enumerate(new_img_list):
                    n_file = os.path.join(c_save_dir, f'{img_names[nx]}_{key}_{ax}{args.dst_suffix}')
                    cv2.imwrite(n_file, n_img)