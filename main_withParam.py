import argparse
from aug_preprocess.aug_for_cls_withParam import alb_aug_cls
from aug_preprocess.aug_for_odt_withParam import alb_aug_odt
# from aug_preprocess.aug_for_msk import imgaug_msk, alb_aug_msk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/MSTAR', help='classification  test')
    parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/COCO', help='object detection val20217')
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/Lunar', help='segmentation train')
    parser.add_argument('--split', type=str, default='val2017', help='[test, val2017, train]')
    parser.add_argument('--task', type=str, default='odt', help='[cls, odt, seg]')
    parser.add_argument('--img_suffix', type=str, default='.jpg', help='.png .jpg')
    parser.add_argument('--lbl_suffix', type=str, default='.txt', help='.txt')
    parser.add_argument('--lbl_format', type=str, default='default', help='coco, voc, yolo, default')
    args = parser.parse_args()

    # use ImgAug
    # if args.task == 'cls':
    #     imgaug_cls(args.base_dir, args.split)
    # elif args.task == 'odt':
    #     imgaug_odt(args.base_dir, args.split)
    # elif args.task == 'seg':
    #     imgaug_msk(args.base_dir, args.split)
    # else:
    #     print('please input task!!!')

    '''
        'scale': {'direction':'up'}, # 'down'
        'translate':{'direction':'left'}, # 'right'
        'crop':{'degree':'low'}, # 'high'
        'flip': {'direction':'H'}, #  'V'
        'rotate': {'direction':'right'}, # 'left'
        'brightness': {'degree':'low'}, # 'high'
        'contrast': {'degree':'low'}, # 'high'
        'saturation': {'degree':'low'}, # 'high'
        'hue': {'degree':'low'}, # 'high'
        'gaussain_noise':{'degree':'low'}, # 'high'
        'poisson_noise':{'degree':'low'}, # 'high'
        'saltpepper':{'degree':'low'}, # 'high'
        'gaussain_blur':{'degree':'low'}, # 'high'
        'mean_blur': {'degree':'low'}, # 'high'
        'fog':{'degree':'low'}, # 'high'
        'rain':{'degree':'low'}, # 'high'
        'snow':{'degree':'low'}, # 'high'
        
        'shear':{'direction':'right'}, # 'left'
        'clahe':{'degree':'low'}, # 'high'
        'defocus':{'degree':'low'}, # 'high'
        'glassblur':{'degree':'low'}, # 'high'
        'multicative_noise':{'degree':'low'}, # 'high'
        'illumination':{'degree':'low'}, # 'high'
        'shadow':{'degree':'low'}, # 'high'
        'posterize':{'degree':'low'}, # 'high'
        'sun':{'degree':'low'}, # 'high'
        'beta_noise':{'degree':'low'}, # 'high'
        'rain_spatter':{'degree':'low'}, # 'high'
        'mud_spatter':{'degree':'low'}, # 'high'
        'random_gravel':{'degree':'low'}, # 'high'
        'super_pixels':{'degree':'low'}, # 'high'
        'sepia':{}, # None
        'deformation':{'degree':'low'}, # 'high'
    '''
    # use Albumentations
    method_params = {
        'rotate': {'direction':'right'},
        'flip': {'direction':'H'},
        'scale': {'direction':'up'}
    }
    if args.task == 'cls':
        alb_aug_cls(args.base_dir, method_params=method_params, exec_num=2, dst_img_suffix=args.img_suffix)
    elif args.task == 'odt':
        alb_aug_odt(args.base_dir, args.split, lbl_format=args.lbl_format, method_params=method_params, exec_num=2, dst_img_suffix=args.img_suffix, dst_lbl_suffix=args.lbl_suffix)
    # elif args.task == 'seg':
    #     alb_aug_msk(args.base_dir, args.split)
    else:
        print('please input task!!!')
