import argparse
from aug_preprocess.aug_for_cls import imgaug_cls, alb_aug_cls
from aug_preprocess.aug_for_odt import imgaug_odt, alb_aug_odt
from aug_preprocess.aug_for_msk import imgaug_msk, alb_aug_msk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/MSTAR', help='classification  test')
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/COCO', help='object detection val20217')
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/Lunar', help='segmentation train')
    parser.add_argument('--split', type=str, default='test', help='[test, val2017, train]')
    parser.add_argument('--task', type=str, default='cls', help='[cls, odt, seg]')
    parser.add_argument('--img_suffix', type=str, default='.jpg', help='.png .jpg')
    parser.add_argument('--lbl_suffix', type=str, default='.txt', help='.txt')
    args = parser.parse_args()

    # use ImgAug
    if args.task == 'cls':
        imgaug_cls(args.base_dir, args.split)
    # elif args.task == 'odt':
    #     imgaug_odt(args.base_dir, args.split)
    # elif args.task == 'seg':
    #     imgaug_msk(args.base_dir, args.split)
    # else:
    #     print('please input task!!!')

    # use Albumentations
    # if args.task == 'cls':
    #     alb_aug_cls(args.base_dir, args.split)
    # elif args.task == 'odt':
    #     alb_aug_odt(args.base_dir, args.split)
    # elif args.task == 'seg':
    #     alb_aug_msk(args.base_dir, args.split)
    # else:
    #     print('please input task!!!')
