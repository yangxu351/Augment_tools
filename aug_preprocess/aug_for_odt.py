import os
from PIL import Image
import pandas as pd
import numpy as np 
from aug_preprocess.utils import make_dir_if_not_exist


def covert_yolo_2_xyxy(lbl, w, h):
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


def imgaug_odt(base_dir, split='val', dst_img_suffix='.jpg', dst_lbl_suffix='.txt'):
    import imgaug as ia 
    from img_imgaug import imgbbx_aug as imgaug
    
    src_img_dir = os.path.join(base_dir, split, 'images')
    src_lbl_dir = os.path.join(base_dir, split, 'labels')
    save_img_dir = os.path.join(base_dir, split + '_imgaug', 'images')
    make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = os.path.join(base_dir, split + '_imgaug', 'labels')
    make_dir_if_not_exist(save_lbl_dir, rm=True)


    ori_imgs = os.listdir(src_img_dir)
    ori_imgs.sort()
    names = [os.path.basename(om).split('.')[0] for om in ori_imgs]
    img_list = []
    lbl_list = []
    for ix, f in enumerate(ori_imgs):
        img = np.array(Image.open(os.path.join(src_img_dir,f)))
        print(' ori shape',  img.shape) # h,w,c
        h, w, c = img.shape
        img_list.append(img)
        
        lbl_file = os.path.join(src_lbl_dir, f'{names[ix]}{dst_lbl_suffix}')
        # print('--lbl---', lbl_file)
        lbl = pd.read_csv(lbl_file, header=None, delimiter=' ')
        arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id = covert_yolo_2_xyxy(lbl, w, h)
        arr_coor = np.array(list(zip(arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id)))
        # print('arr lwh rwh', arr_coor[:, :])

        bbx_list = []
        for j in range(arr_coor.shape[0]):
            bbx = ia.BoundingBox(*arr_coor[j, :5])
            bbx_list.append(bbx)
        bbxoi = ia.BoundingBoxesOnImage(bbx_list, shape=img.shape)
        lbl_list.append(bbxoi)
        # ia.imshow(bbxoi.draw_on_image(img))

    dict_trans = imgaug.get_trans()
    
    for key, trans in dict_trans.items():
        for ax, aug in enumerate(trans):
            for nx, n_img in enumerate(img_list):
                new_img, new_bbx = aug(image=n_img, bounding_boxes=lbl_list[nx])
                # ia.imshow(new_bbx.draw_on_image(new_img))
                n_img_file = os.path.join(save_img_dir, f'{names[nx]}_{key}_{ax}{dst_img_suffix}')
                Image.fromarray(new_img).save(n_img_file)
                n_lbl_file = os.path.join(save_lbl_dir, f'{names[nx]}_{key}_{ax}{dst_lbl_suffix}')
                with open(n_lbl_file, 'w') as f:
                    for ann in new_bbx:
                        f.write("%s %s %s %s %s\n" % (int(ann.label), ann.x1, ann.y1, ann.x2, ann.y2))
                f.close()

    

def alb_aug_odt(base_dir, split='val', dst_img_suffix='.jpg', dst_lbl_suffix='.txt'):
    from img_albumentation import imgcls_aug as albaug

    src_img_dir = os.path.join(base_dir, split, 'images')
    src_lbl_dir = os.path.join(base_dir, split, 'labels')
    save_img_dir = os.path.join(base_dir, split + '_albaug', 'images')
    make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = os.path.join(base_dir, split + '_albaug', 'labels')
    make_dir_if_not_exist(save_lbl_dir, rm=True)


    ori_imgs = os.listdir(src_img_dir)
    ori_imgs.sort()
    names = [os.path.basename(om).split('.')[0] for om in ori_imgs]
    img_list = []
    for ix, f in enumerate(ori_imgs):
        img = np.array(Image.open(os.path.join(src_img_dir, f)))
        print(' ori shape',  img.shape) # h,w,c
        h, w, c = img.shape
        img_list.append(img)
        lbl_file = os.path.join(src_lbl_dir, f'{names[ix]}{dst_lbl_suffix}')
        print('--lbl---', lbl_file)
        lbl = pd.read_csv(lbl_file, header=None, delimiter=' ')
        arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id = covert_yolo_2_xyxy(lbl, w, h)
        arr_coor =np.array(list(zip(arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id)))
        # list_cid = arr_id.tolist()
        # print('arr lwh rwh', arr_coor[:, :])

        # bbx_list = []
        # for j in range(arr_coor.shape[0]):
        #     bbx = ia.BoundingBox(*arr_coor[j, :5])
        #     bbx_list.append(bbx)
        # bbxoi = ia.BoundingBoxesOnImage(bbx_list, shape=img.shape)
        # lbl_list.append(bbxoi)
        # ia.imshow(bbxoi.draw_on_image(img))

    dict_trans = albaug.get_trans()
    
    for key, trans in dict_trans.items():
        for ax, aug in enumerate(trans):
            for nx, n_img in enumerate(img_list):
                dict_img_bbx = aug(image=n_img, bboxes=arr_coor, bbox_classes=arr_id)
                new_img = dict_img_bbx['image']
                new_bbx = dict_img_bbx['bboxes']
                new_bbx_cid = dict_img_bbx['bbox_classes']
                # ia.imshow(new_bbx.draw_on_image(new_img))
                n_img_file = os.path.join(save_img_dir, f'{names[nx]}_{key}_{ax}{dst_img_suffix}')
                Image.fromarray(new_img).save(n_img_file)
                n_lbl_file = os.path.join(save_lbl_dir, f'{names[nx]}_{key}_{ax}{dst_lbl_suffix}')
                with open(n_lbl_file, 'w') as f:
                    for jx, ann in enumerate(new_bbx):
                        f.write("%s %s %s %s %s\n" % (int(new_bbx_cid[jx]), ann[0], ann[1], ann[2], ann[3]))
                f.close()
    
