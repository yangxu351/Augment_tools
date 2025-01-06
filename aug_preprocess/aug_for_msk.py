import os
from PIL import Image
import numpy as np 
from aug_preprocess.utils import make_dir_if_not_exist


def imgaug_msk(base_dir, split='val', dst_suffix='.jpg'):
    from img_imgaug import imgmsk_aug as imgaug
    import imgaug.augmentables.segmaps as ias

    src_img_dir = os.path.join(base_dir, split, 'images')
    src_msk_dir = os.path.join(base_dir, split, 'masks')
    save_img_dir = os.path.join(base_dir, split + '_imgaug', 'images')
    make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = os.path.join(base_dir, split + '_imgaug', 'masks')
    make_dir_if_not_exist(save_lbl_dir, rm=True)

    img_names = os.listdir(src_img_dir)
    img_names.sort()
    msk_names = os.listdir(src_msk_dir)
    msk_names.sort()
    assert len(img_names)==len(msk_names), 'the quantity of images is different from that of masks'
    img_list = []
    map_list = []
    for ix in range(len(img_names)):# 
        name = img_names[ix]
        img = np.array(Image.open(os.path.join(src_img_dir, name)))
        print(' ori shape',  img.shape) # h,w,c
        img_list.append(img)
        
        msk_file = os.path.join(src_msk_dir, f'{msk_names[ix]}')
        
        # print('--msk---', msk_file)
        msk = np.array(Image.open(msk_file))
        segmap = ias.SegmentationMapsOnImage(msk, shape=img.shape)
        map_list.append(segmap)
        # map_list.append(np.expand_dims(msk, axis=0).astype(np.int32))

    dict_trans = imgaug.get_trans()
    
    trans_indexes= ['ari_dr_aug','ari_ns_aug','ari_wh_aug','bld_aug','blu_aug','clr_aug','wea_aug']
    for key, trans in dict_trans.items():
        if key not in trans_indexes:
            continue
        for ax, aug in enumerate(trans):
            for nx, n_img in enumerate(img_list):
                new_img, new_msk = aug(image=n_img, segmentation_maps=map_list[nx])
                n_img_file = os.path.join(save_img_dir, f'{img_names[nx]}_{key}_{ax}{dst_suffix}')
                Image.fromarray(new_img).save(n_img_file)
                n_msk_file = os.path.join(save_lbl_dir, f'{msk_names[nx]}_{key}_{ax}{dst_suffix}')
                Image.fromarray(msk+100).save(n_msk_file) # mask 不变
                # Image.fromarray(new_msk.get_arr(), mode='L').save(n_msk_file)
                # Image.fromarray(new_msk[:,:,0]).save(n_msk_file)

    
    
def alb_aug_msk(base_dir, split='val', dst_suffix='.jpg'):
    from img_albumentation import imgcls_aug as albaug

    src_img_dir = os.path.join(base_dir, split, 'images')
    src_msk_dir = os.path.join(base_dir, split, 'masks')
    save_img_dir = os.path.join(base_dir, split + '_albaug', 'images')
    make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = os.path.join(base_dir,split  + '_albaug', 'masks')
    make_dir_if_not_exist(save_lbl_dir, rm=True)

    img_names = os.listdir(src_img_dir)
    img_names.sort()
    msk_names = os.listdir(src_msk_dir)
    msk_names.sort()
    assert len(img_names)==len(msk_names), 'the quantity of images is different from that of masks'
    img_list = []
    map_list = []
    for ix in range(len(img_names)):# len(img_names)
        name = img_names[ix]
        img = np.array(Image.open(os.path.join(src_img_dir, name)))
        # print(' ori shape',  img.shape) # h,w,c
        img_list.append(img)
        
        msk_file = os.path.join(src_msk_dir, f'{msk_names[ix]}')
        
        # print('--msk---', msk_file)
        msk = np.array(Image.open(msk_file))
        map_list.append(msk)

    dict_trans = albaug.get_trans()
    trans_indexes= ['ari_dr_aug','ari_ns_aug','ari_wh_aug','blu_aug','clr_aug','wea_aug']
    
    for key, trans in dict_trans.items():
        if key not in trans_indexes:
            continue
        for ax, aug in enumerate(trans):
            for nx, n_img in enumerate(img_list):
                dict_img_msk = aug(image=n_img, mask=map_list[nx])
                new_img = dict_img_msk["image"]
                new_msk = dict_img_msk["mask"]
                n_img_file = os.path.join(save_img_dir, f'{img_names[nx]}_{key}_{ax}{dst_suffix}')
                Image.fromarray(new_img).save(n_img_file)
                n_msk_file = os.path.join(save_lbl_dir, f'{msk_names[nx]}_{key}_{ax}{dst_suffix}')
                Image.fromarray(msk+100).save(n_msk_file)# mask 不变