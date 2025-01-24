import argparse
from aug_preprocess.aug_for_cls_withParam import alb_aug_cls
from aug_preprocess.aug_for_odt_withParam import alb_aug_odt
from aug_preprocess.aug_for_msk_withParam import alb_aug_msk
import os 
import warnings
import json
import logging
# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_augmentation.log'
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    warnings.filterwarnings(action='ignore', category=UserWarning)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/MSTAR', help='classification  test')
    # parser.add_argument('--split', type=str, default='test', help='[test, val2017, train]')
    # parser.add_argument('--task', type=str, default='cls', help='[cls, odt, seg]')

    # parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/COCO', help='object detection val20217')        
    # parser.add_argument('--split', type=str, default='val2017', help='[test, val2017, train]')
    # parser.add_argument('--task', type=str, default='odt', help='[cls, odt, seg]')
    # parser.add_argument('--lbl_format', type=str, default='coco', help='coco, voc, yolo, default')

    parser.add_argument('--base_dir', type=str, default='F:/Public_Dataset/ForAug/Lunar', help='segmentation train')
    parser.add_argument('--task', type=str, default='seg', help='[cls, odt, seg]')
    parser.add_argument('--split', type=str, default=['train'], help='[test, train]')
    parser.add_argument('--odt_lbl_format', type=str, default='labelme', help='[labelme, voc, yolo, coco]')
    parser.add_argument('--exec_num', type=int, default=1, help='数据增强次数')

    parser.add_argument('--method_params', type=str, default="\{'scale':{'direction':'up'}\}", help='')


    args = parser.parse_args()
    params_file = os.path.join(args.base_dir, 'received_params', 'data_aug_params.json')
    dict_params = json.load(open(params_file))
    args.task = dict_params['task']
    args.split = dict_params['split']
    if args.task == 'odt':
        args.odt_lbl_format = dict_params['odt_lbl_format']
    args.exec_num = dict_params['exec_num']
    args.method_params = dict_params['method_params']

    method_params = json.loads(args.method_params)
    
    # use Albumentations
    method_params = {
        ## 几何变换
        'scale': {'direction':'zoonin'}, # 'zoomout'-------------
        'translate':{'direction':'left'}, # 'right'-------------
        'crop':{'degree':'low'}, # 'high'
        'flip': {'direction':'H'}, #  'V'-------------
        'rotate': {'direction':'right'}, # 'left'-------------
        ## 颜色变换
        'brightness': {'degree':'low'}, # 'high'
        'contrast': {'degree':'low'}, # 'high'
        'saturation': {'degree':'low'}, # 'high'
        'hue': {'degree':'low'}, # 'high'
        ## 加噪声
        'gaussain_noise':{'degree':'low'}, # 'high'
        'poisson_noise':{'degree':'low'}, # 'high'
        'saltpepper':{'degree':'low'}, # 'high'
        ## 模糊
        'gaussain_blur':{'degree':'low'}, # 'high'
        'mean_blur': {'degree':'low'}, # 'high'
        ## 天气变化
        'fog':{'degree':'low'}, # 'high'
        'rain':{'degree':'light'}, # 'medium' 'heavy'-------------
        'snow':{'degree':'low'}, # 'high'
        'shear':{'direction':'left'}, # 'right'------------- # 错切变换
        'clahe':{'degree':'low'}, # 'high' # 直方图均衡
        'defocus':{'degree':'low'}, # 'high' # 散焦模糊
        'glassblur':{'degree':'low'}, # 'high'
        'multicative_noise':{'degree':'low'}, # 'high'
        'illumination':{'degree':'cool'}, # 'warm' -------------# 照明变换
        'shadow':{'degree':'low'}, # 'high'
        'posterize':{'degree':'low'}, # 'high'  #颜色通道位数变换
        'sun':{'degree':'low'}, # 'high'        #耀斑
        'overshoot':{'degree':'low'}, # 'high'  #随机伪影
        'rain_spatter':{'degree':'low'}, # 'high' # 雨飞溅
        'mud_spatter':{'degree':'low'}, # 'high'  # 泥飞溅
        'random_gravel':{'degree':'low'}, # 'high' # 随机砾石
        'super_pixels':{'degree':'low'}, # 'high'
        'sepia':{}, # None # 棕褐色滤镜------------
        'deformation':{'degree':'low'}, # 'high' # 非刚性形变

    }
    
    if args.task == 'cls':
        logger.info('开始图像分类任务的数据增强-----')
        alb_aug_cls(args.base_dir, split=args.split, method_params=args.method_params, exec_num=args.exec_num)
    elif args.task == 'odt':
        logger.info('开始目标检测任务的数据增强-----')
        alb_aug_odt(args.base_dir, split=args.split, lbl_format=args.odt_lbl_format, method_params=args.method_params, exec_num=args.exec_num)
    elif args.task == 'seg':
        logger.info('开始语义分割任务的数据增强-----')
        alb_aug_msk(args.base_dir, split=args.split, method_params=args.method_params, exec_num=args.exec_num)
    else:
        logger.error('please input task!!!')
    
    logger.info('---------数据增强结束！')

        # ## 几何变换
        # 'scale': {'direction':'zoonin'}, # 'zoomout'-------------
        # 'translate':{'direction':'left'}, # 'right'-------------
        # 'crop':{'degree':'low'}, # 'high'
        # 'flip': {'direction':'H'}, #  'V'-------------
        # 'rotate': {'direction':'right'}, # 'left'-------------
        # 'shear':{'direction':'left'}, # 'right'------------- # 错切变换
        # 'deformation':{'degree':'low'}, # 'high' # 非刚性形变
        # ## 颜色变换
        # 'brightness': {'degree':'low'}, # 'high'
        # 'contrast': {'degree':'low'}, # 'high'
        # 'saturation': {'degree':'low'}, # 'high'
        # 'hue': {'degree':'low'}, # 'high'
        # 'clahe':{'degree':'low'}, # 'high' # 直方图均衡
        # 'posterize':{'degree':'low'}, # 'high'  #颜色通道位数变换
        # 'sepia':{}, # None # 棕褐色滤镜------------
        # 'illumination':{'degree':'cool'}, # 'warm' -------------# 照明变换
       

        # ## 加噪声
        # 'gaussain_noise':{'degree':'low'}, # 'high'
        # 'poisson_noise':{'degree':'low'}, # 'high'
        # 'saltpepper':{'degree':'low'}, # 'high'
        # 'shadow':{'degree':'low'}, # 'high'
        # 'random_gravel':{'degree':'low'}, # 'high' # 随机砾石
        # 'rain_spatter':{'degree':'low'}, # 'high' # 雨飞溅
        # 'mud_spatter':{'degree':'low'}, # 'high'  # 泥飞溅

        # ## 模糊
        # 'gaussain_blur':{'degree':'low'}, # 'high'
        # 'mean_blur': {'degree':'low'}, # 'high'
        # 'defocus':{'degree':'low'}, # 'high' # 散焦模糊
        # 'glassblur':{'degree':'low'}, # 'high'
        # 'multicative_noise':{'degree':'low'}, # 'high'
        # 'super_pixels':{'degree':'low'}, # 'high'
        # 'overshoot':{'degree':'low'}, # 'high'  #随机伪影
        

        # ## 天气变化
        # 'fog':{'degree':'low'}, # 'high'
        # 'rain':{'degree':'light'}, # 'medium' 'heavy'-------------
        # 'snow':{'degree':'low'}, # 'high'
        # 'sun':{'degree':'low'}, # 'high'        #耀斑

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        