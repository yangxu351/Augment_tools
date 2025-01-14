import os
from PIL import Image
import pandas as pd
import numpy as np 
from aug_preprocess.utils import make_dir_if_not_exist
from img_albumentation import img_aug_param as albaug
import datetime
import albumentations as alb

def convert_id_last(lbl): # default
    # return  (normalized) [xmin ymin xmax ymax ID] [0.153125 0.71875 0.65625 0.9625 2]
    arr_lbl = lbl.to_numpy()
    arr_tl_w = arr_lbl[:, 1]
    arr_tl_h = arr_lbl[:, 2]
    arr_br_w = arr_lbl[:, 3]
    arr_br_h = arr_lbl[:, 4]
    arr_id = arr_lbl[:, 0]
    arr_coor =np.array(list(zip(arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id)))
    return arr_coor

def covert_yolo_2_default(lbl):
    #YOLO: (normalized) [ID center_x center_y width height] [2 0.4046875 0.840625 0.503125 0.24375]
    arr_lbl = lbl.to_numpy()
    arr_id, center_x, center_y, width, height = arr_lbl[:,0], arr_lbl[:,1], arr_lbl[:,2], arr_lbl[:,3], arr_lbl[:,4]
    # print('ori arr', arr_lbl)
    arr_tl_w = np.round(center_x - width / 2, decimals=4)
    arr_tl_h = np.round(center_y - height / 2, decimals=4)
    arr_br_w = np.round(center_x + width / 2, decimals=4)
    arr_br_h = np.round(center_y + height / 2, decimals=4)
    arr_coor =np.array(list(zip(arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id)))
    return arr_coor

def covert_voc_2_default(lbl, w, h):
    # VOC:  (unnormalized) [ID xmin ymin xmax ymax] [2 98 345 420 462]
    lbl.iloc[:, 1] /= w
    lbl.iloc[:, 2] /= h
    lbl.iloc[:, 3] /= w
    lbl.iloc[:, 4] /= h
    arr_lbl = lbl.to_numpy()
    arr_id = arr_lbl[:, 0]
    arr_tl_w = np.round(arr_lbl[:,1], decimals=4)
    arr_tl_h = np.round(arr_lbl[:,2], decimals=4)
    arr_br_w = np.round(arr_lbl[:,1], decimals=4)
    arr_br_h = np.round(arr_lbl[:,2], decimals=4)
    arr_coor =np.array(list(zip(arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id)))
    return arr_coor

def covert_coco_2_default(lbl, w, h):
    # COCO: (unnormalized) [ID center_x center_y width height] [2 98 345 322 117]
    lbl.iloc[:, 1] /= w
    lbl.iloc[:, 2] /= h
    lbl.iloc[:, 3] /= w
    lbl.iloc[:, 4] /= h
    arr_lbl = lbl.to_numpy()
    arr_id, center_x, center_y, width, height = arr_lbl[:,0], arr_lbl[:,1], arr_lbl[:,2], arr_lbl[:,3], arr_lbl[:,4]
    arr_tl_w = np.round(center_x - width / 2, decimals=4)
    arr_tl_h = np.round(center_y - height / 2, decimals=4)
    arr_br_w = np.round(center_x + width / 2, decimals=4)
    arr_br_h = np.round(center_y + height / 2, decimals=4)
    arr_coor =np.array(list(zip(arr_tl_w, arr_tl_h, arr_br_w, arr_br_h, arr_id)))
    return arr_coor


def alb_aug_odt(base_dir, split='val', lbl_format='default', method_params={}, combine=False, exec_num=1):
    '''
    lbl_format: 'coco', 'pascal_voc', 'albumentations', 'yolo'
    '''
    if lbl_format=='default':
        lbl_format='albumentations'
    elif lbl_format=='voc':
        lbl_format='pascal_voc'

    src_img_dir = os.path.join(base_dir, split, 'images')
    src_lbl_dir = os.path.join(base_dir, split, 'labels')
    save_img_dir = os.path.join(base_dir, split + '_albaug_param', 'images')
    # make_dir_if_not_exist(save_img_dir, rm=True)
    save_lbl_dir = os.path.join(base_dir, split + '_albaug_param', 'labels')
    # make_dir_if_not_exist(save_lbl_dir, rm=True)

    ori_img_names = os.listdir(src_img_dir)
    ori_img_names.sort()
    ori_lbl_names = os.listdir(src_lbl_dir)
    ori_lbl_names.sort()
    # names = [os.path.basename(om).split('.')[0] for om in ori_img_names]

    all_trans = albaug.get_trans()
    dict_trans = {}
    for method, params in method_params.items():
        dict_trans[method] = all_trans[method](**params)


    for ix, img_name in enumerate(ori_img_names[:2]):
        img = np.array(Image.open(os.path.join(src_img_dir, img_name)))
        img_name_pref, img_name_suff = img_name.split('.')
        print(' ori shape',  img.shape) # h,w,c
        h, w, c = img.shape
        # img_list.append(img)
        lbl_file = os.path.join(src_lbl_dir, f'{ori_lbl_names[ix]}')
        lbl_name_pref, lbl_name_suff = ori_lbl_names[ix].split('.')
        print('--lbl---', lbl_file)
        lbl = pd.read_csv(lbl_file, header=None, delimiter=' ')
        
        arr_coor = lbl.iloc[:,1:5].to_numpy()
        arr_ids = lbl.iloc[:, 0].to_numpy()
        
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%m%d-%H%M")
        for key, aug in dict_trans.items():
            o_save_img_dir = os.path.join(save_img_dir, key)
            make_dir_if_not_exist(o_save_img_dir, rm=True)
            o_save_lbl_dir = os.path.join(save_lbl_dir, key)
            make_dir_if_not_exist(o_save_lbl_dir, rm=True)
            for en in range(exec_num):
            
                transform = alb.Compose([aug], bbox_params=alb.BboxParams(format=lbl_format, label_fields=['category_ids']))
                dict_img_bbx = transform(image=img, bboxes=arr_coor, category_ids=arr_ids)
                new_img = dict_img_bbx['image']
                new_bbx = dict_img_bbx['bboxes']
                cids = dict_img_bbx['category_ids']
                n_img_file = os.path.join(o_save_img_dir, f'{img_name_pref}_ex{en+1}_{formatted_time}.{img_name_suff}')
                Image.fromarray(new_img).save(n_img_file)
                n_lbl_file = os.path.join(o_save_lbl_dir, f'{lbl_name_pref}_ex{en+1}_{formatted_time}.{lbl_name_suff}')
                with open(n_lbl_file, 'w') as img_name:
                    for jx, ann in enumerate(new_bbx):
                        img_name.write("%s %s %s %s %s\n" % (int(cids[jx]), round(ann[0],4), round(ann[1],4), round(ann[2],4), round(ann[3],4)))
                img_name.close()

        # if lbl_format=='yolo': # (normalized) [ID center_x center_y width height] [2 0.4046875 0.840625 0.503125 0.24375]
        #     arr_coor = covert_yolo_2_default(lbl, w, h)
        # elif lbl_format=='voc': # (unnormalized) [ID xmin ymin xmax ymax] [2 98 345 420 462]
        #     arr_coor = covert_voc_2_default(lbl, w, h)
        # elif lbl_format=='coco': # (unnormalized) [ID center_x center_y width height] [2 98 345 322 117]
        #     arr_coor = covert_coco_2_default(lbl, w, h)
        # else: # 'default' (normalized) [ID xmin ymin xmax ymax] [2 0.153125 0.71875 0.65625 0.9625]
        #     arr_coor = convert_id_last(lbl)

        # for key, aug in dict_trans.items():
        #     o_save_img_dir = os.path.join(save_img_dir, key)
        #     make_dir_if_not_exist(o_save_img_dir, rm=True)
        #     o_save_lbl_dir = os.path.join(save_lbl_dir, key)
        #     make_dir_if_not_exist(o_save_lbl_dir, rm=True)
        #     current_time = datetime.datetime.now()
        #     formatted_time = current_time.strftime("%m%d-%H%M")
            
        #     for en in range(exec_num):
        #         # dict_img_bbx = aug(image=img, bboxes=arr_coor[:,:4], bbox_classes=arr_coor[:,4])
        #         new_img = dict_img_bbx['image']
        #         new_bbx = dict_img_bbx['bboxes']
        #         new_bbx_cid = dict_img_bbx['bbox_classes']
        #         # ia.imshow(new_bbx.draw_on_image(new_img))
        #         n_img_file = os.path.join(o_save_img_dir, f'{names[ix]}_ex{en+1}_{formatted_time}{dst_img_suffix}')
        #         Image.fromarray(new_img).save(n_img_file)
        #         n_lbl_file = os.path.join(o_save_lbl_dir, f'{names[ix]}_ex{en+1}_{formatted_time}{dst_lbl_suffix}')
        #         with open(n_lbl_file, 'w') as f:
        #             for jx, ann in enumerate(new_bbx):
        #                 f.write("%s %s %s %s %s\n" % (int(new_bbx_cid[jx]), round(ann[0],4), round(ann[1],4), round(ann[2],4), round(ann[3],4)))
        #         f.close()

