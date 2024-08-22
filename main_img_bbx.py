import os
import shutil
import argparse
import ast
import glob
from PIL import Image
import pandas as pd
import numpy as np 
from img_imgaug.imgbbx_aug import *


def make_dir_if_not_exist(mf, rm=False):
    if not os.path.exists(mf):
        os.makedirs(mf)
    else:
        if rm: # remove folder if exists
            shutil.rmtree(mf)
            os.mkdir(mf)


def covert_yolo_2_xyxy(lbl):
    lbl.iloc[:, 1] *= w
    lbl.iloc[:, 2] *= h
    lbl.iloc[:, 3] *= w
    lbl.iloc[:, 4] *= h
    arr_lbl = lbl.to_numpy()
    # print('ori arr', arr_lbl)
    arr_id = arr_lbl[:, 0]
    half_w = arr_lbl[:,3]/2.
    half_h = arr_lbl[:,4]/2.
    arr_tl_w = np.round(arr_lbl[:,1] - half_w, decimals=2)
    arr_tl_h = np.round(arr_lbl[:,2] - half_h, decimals=2)
    arr_br_w = np.round(arr_lbl[:,1] + half_w, decimals=2)
    arr_br_h = np.round(arr_lbl[:,2] + half_h, decimals=2)
    return arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/COCO/val2017')
    parser.add_argument('--aug_list', type=ast.literal_eval, 
                        default="['art_cartoon', 'ari_drop', 'ari_noise', 'ari_while', 'blend', 'blur', 'color', 'geometric', 'weather', 'meta']", 
                        help='art')
    parser.add_argument('--img_suffix', type=str, default='.jpg', help='.png .jpg')
    parser.add_argument('--lbl_suffix', type=str, default='.txt', help='.txt')
    args = parser.parse_args()
    
    src_img_dir = os.path.join(args.base_dir, 'images')
    src_lbl_dir = os.path.join(args.base_dir, 'labels')
    save_img_dir = src_img_dir + '_aug'
    make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = src_lbl_dir + '_aug'
    make_dir_if_not_exist(save_lbl_dir, rm=True)


    ori_imgs = glob.glob(os.path.join(src_img_dir, f'*{args.img_suffix}'))
    ori_imgs.sort()
    names = [os.path.basename(om).split('.')[0] for om in ori_imgs]
    img_list = []
    lbl_list = []
    for ix, f in enumerate(ori_imgs[:2]):
        img = np.array(Image.open(f))
        print(' ori shape',  img.shape) # h,w,c
        h, w, c = img.shape
        img_list.append(img)
        
        lbl_file = os.path.join(src_lbl_dir, f'{names[ix]}{args.lbl_suffix}')
        print('--lbl---', lbl_file)
        lbl = pd.read_csv(lbl_file, header=None, delimiter=' ')
        arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id = covert_yolo_2_xyxy(lbl)
        arr_coor = np.array(list(zip(arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id)))
        print('arr lwh rwh', arr_coor[:, :])

        bbx_list = []
        for j in range(arr_coor.shape[0]):
            bbx = ia.BoundingBox(*arr_coor[j, :5])
            bbx_list.append(bbx)
        bbxoi = ia.BoundingBoxesOnImage(bbx_list, shape=img.shape)
        lbl_list.append(bbxoi)
        # ia.imshow(bbxoi.draw_on_image(img))

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
    
    for key, trans in dict_trans.items():
        for ax, aug in enumerate(trans):
            for nx, n_img in enumerate(img_list):
                new_img, new_bbx = aug(image=n_img, bounding_boxes=lbl_list[nx])
                # ia.imshow(new_bbx.draw_on_image(new_img))
                n_img_file = os.path.join(save_img_dir, f'{names[nx]}_{key}_{ax}{args.img_suffix}')
                Image.fromarray(new_img).save(n_img_file)
                n_lbl_file = os.path.join(save_lbl_dir, f'{names[nx]}_{key}_{ax}{args.lbl_suffix}')
                with open(n_lbl_file, 'w') as f:
                    for ann in new_bbx:
                        f.write("%s %s %s %s %s\n" % (int(ann.label), ann.x1, ann.y1, ann.x2, ann.y2))
                f.close()
