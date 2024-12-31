import argparse
from aug_preprocess.aug_for_cls import imgaug_cls
from aug_preprocess.aug_for_odt import imgaug_odt
from aug_preprocess.aug_for_msk import imgaug_msk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/MSTAR', help='classification')
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/COCO', help='object detection')
    parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/Lunar', help='segmentation')
    parser.add_argument('--split', type=str, default='val', help='[val, train, test]')
    parser.add_argument('--task', type=str, default='seg', help='[cls, odt, seg]')
    parser.add_argument('--img_suffix', type=str, default='.jpg', help='.png .jpg')
    parser.add_argument('--lbl_suffix', type=str, default='.txt', help='.txt')
    args = parser.parse_args()

    # use ImgAug
    if args.task == 'cls':
        imgaug_cls(args.base_dir, args.split)
    elif args.task == 'odt':
        imgaug_odt(args.base_dir, args.split)
    elif args.task == 'seg':
        imgaug_msk(args.base_dir, args.split)
    else:
        print('please input task!!!')

    # use Albumentations
    
