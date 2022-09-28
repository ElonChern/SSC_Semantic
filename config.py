import numpy as np
import torch

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'nyu':
            # folder that contains dataset/.
            return {'train': '/data/elon/NYU_SSC/NYUtrain_npz',
                    'val': '/data/elon//NYU_SSC/NYUtest_npz'}

        elif dataset == 'nyucad':
            return {'train': '/data/elon/NYU_SSC/NYUCADtrain_npz',
                    'val': '/data/elon/NYU_SSC/NYUCADtest_npz'}

        # debug
        elif dataset == 'debug':
            return {'train': '/home/jsg/jie/Data_zoo/NYU_SSC/NYUCADval40_npz',
                    'val': '/home/jsg/jie/Data_zoo/NYU_SSC/NYUCADval40_npz'}
            
        elif dataset == 'semantic_kitti':
            return {'train': '/data/elon/MMSCNet/semantic_kitti_package',
                    'val': '/data/elon/MMSCNet/semantic_kitti_package'}

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


# ssc: color map

colorMap = np.array([[0, 0, 0],   # 0 "unlabeled", and others ignored 0
                    [100, 150, 245],    # 1 "car" 10
                    [100, 230, 245],      # 2 "bicycle" 11
                    [30, 60, 150],   # 3 "motorcycle" 15 棕色
                    [80, 30, 180],   # 4 "truck" 18 绛红
                    [100, 80, 250],    # 5 "other-vehicle" 20 红色
                    [255, 30, 30],   # 6 "person" 30 淡蓝色
                    [255, 40, 200],   # 7 "bicyclist" 31 淡紫色
                    [150, 30, 90],    # 8 "motorcyclist" 32 深紫色
                    [255, 0, 255],    # 9 "road" 40 浅紫色
                    [255, 150, 255],    # 10 "parking" 44 紫色
                    [75, 0, 75],   # 11 "sidewalk" 48 紫黑色
                    [175, 0, 75],   # 12 "other-ground" 49 深蓝色
                    [255, 200, 0],   # 13 "building" 50 浅蓝色
                    [255, 120, 50],   # 14 "fence" 51 蓝色
                    [0, 175, 0],   # 15 "vegetation" 70 绿色
                    [135, 60, 0],   # 16 "trunk" 71 蓝色
                    [150, 240, 80],   # 17 "terrain" 72 青绿色
                    [255, 240, 150],   # 18 "pole"80 天空蓝
                    [255, 0, 0]   # 19 "traffic-sign" 81 标准蓝
                    ]).astype(np.int32) 
# ###########################################################################################

# class_weights = torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

semantic_kitti_class_frequencies = np.array(
                                    [
                                        5.41773033e09,
                                        1.57835390e07,
                                        1.25136000e05,
                                        1.18809000e05,
                                        6.46799000e05,
                                        8.21951000e05,
                                        2.62978000e05,
                                        2.83696000e05,
                                        2.04750000e05,
                                        6.16887030e07,
                                        4.50296100e06,
                                        4.48836500e07,
                                        2.26992300e06,
                                        5.68402180e07,
                                        1.57196520e07,
                                        1.58442623e08,
                                        2.06162300e06,
                                        3.69705220e07,
                                        1.15198800e06,
                                        3.34146000e05,
                                    ])

class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))
class_weights = class_weights.float()

       