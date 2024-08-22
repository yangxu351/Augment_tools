import os
from PIL import Image
import numpy as np 
from img_imgaug.imgmsk_aug import get_trans
import imgaug.augmentables.segmaps as ias
import imgaug as ia
from utils import make_dir_if_not_exist


def aug_msk(base_dir, dst_suffix='.jpg'):
    src_img_dir = os.path.join(base_dir, 'images')
    src_msk_dir = os.path.join(base_dir, 'masks')
    save_img_dir = os.path.join(base_dir + '_aug', 'images')
    make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = os.path.join(base_dir + '_aug', 'masks')
    make_dir_if_not_exist(save_lbl_dir, rm=True)

    img_names = os.listdir(src_img_dir)
    img_names.sort()
    msk_names = os.listdir(src_msk_dir)
    msk_names.sort()
    assert len(img_names)==len(msk_names), 'the quantity of images is different from that of masks'
    img_list = []
    map_list = []
    for ix in range(3):# len(img_names)
        name = img_names[ix]
        img = np.array(Image.open(os.path.join(src_img_dir, name)))
        print(' ori shape',  img.shape) # h,w,c
        img_list.append(img)
        
        msk_file = os.path.join(src_msk_dir, f'{msk_names[ix]}')
        
        # print('--msk---', msk_file)
        msk = np.array(Image.open(msk_file))
        segmap = ias.SegmentationMapsOnImage(msk, shape=img.shape)
        map_list.append(segmap)

    dict_trans = get_trans()
    
    for key, trans in dict_trans.items():
        for ax, aug in enumerate(trans):
            for nx, n_img in enumerate(img_list):
                new_img, new_msk = aug(image=n_img, segmentation_maps=map_list[nx])
                # ia.imshow(new_bbx.draw_on_image(new_img))
                n_img_file = os.path.join(save_img_dir, f'{img_names[nx]}_{key}_{ax}{dst_suffix}')
                Image.fromarray(new_img).save(n_img_file)
                n_msk_file = os.path.join(save_lbl_dir, f'{msk_names[nx]}_{key}_{ax}{dst_suffix}')
                ia.imshow(new_msk.draw_on_image(new_img)[0])
                Image.fromarray(new_msk.get_arr()).save(n_msk_file)

    
    
