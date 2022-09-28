#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Class of pytorch data loader
---
Jie Li
jieli_cn@163.com
Nanjing University of Science and Technology
Aug 10, 2019
"""

import glob
import imageio
import numpy as np
import numpy.matlib
import torch.utils.data
import os
from torchvision import transforms
import sys
import yaml
from PIL import Image
import cv2
sys.path.append('/home/elon/SSC-Semantic/')
from config import colorMap
from torchvision import transforms
from dataloaders import io_data as SemanticKittiIO
from dataloaders.utils.helpers import vox2pix


class SemanticKittiDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase,istest=False):

        self.scene_size = (51.2, 51.2, 6.4)
       
        self.root = root
        yaml_path,_ = os.path.split(os.path.realpath(__file__))
           
        sequence = {"train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
                  "val": ["08"],
                  "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],}
        self.phase = phase
        self.dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'semantic-kitti.yaml'),'r'))
        self.nbr_classes = self.dataset_config['nbr_classes']
        self.grid_dimensions = self.dataset_config['grid_dims'] # [W, H, D] [256,32,256]
        self.sequences = sequence[self.phase]        
        self.remap_lut = self.get_remap_lut()
        
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1220
        self.img_H = 370
        
        self.istest = istest
       
        self.filepaths = self.get_filelist(self.root, self.sequences)

        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] \
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        self.transforms_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.34749558, 0.36745213, 0.36123651], std=[0.30599035, 0.3129534 , 0.31933814]),
        ])
        print('Dataset:{} files'.format(len(self.filepaths)))

    def __getitem__(self, index):
        """
        Returns:
            key = ['img','depth','target_1_1','target_1_2','target_1_4','target_1_8','occupancy',
                'fov_target_1_1','fov_target_1_2','fov_target_1_4','fov_target_1_8','occluded','invalid,
                'projected_pix_1','projected_pix_2','projected_pix_4','projected_pix_8',
                "fov_mask_1","fov_mask_2","fov_mask_4","fov_mask_8"]
            sequence, frame_id 
        """
                
        data = {}
        filepath = self.filepaths[index]
        sequence = filepath["sequence"]
        P = filepath["P"]
        cam_k = P[0:3, 0:3]
        T_velo_2_cam = filepath["T_velo_2_cam"]
        voxel_path = filepath["voxel_path"]
        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]
    
        scale_3ds = [1, 2, 4, 8]
        for scale_3d in scale_3ds:

            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(T_velo_2_cam,
                                                        cam_k,
                                                        self.vox_origin,
                                                        self.voxel_size * scale_3d,
                                                        self.img_W,
                                                        self.img_H,
                                                        self.scene_size,
                                                    )            

            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["pix_z_{}".format(scale_3d)] = pix_z
            data["fov_mask_{}".format(scale_3d)] = fov_mask        

        
        # voxel
        occupancy = SemanticKittiIO._read_occupancy_SemKITTI(voxel_path)
        occupancy = np.moveaxis(occupancy.reshape([self.grid_dimensions[0],
                                                    self.grid_dimensions[2],
                                                    self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
        occupancy = occupancy[None,:,:,:]
        data['occupancy'] = occupancy
        
        #  occluded
        occluded_path = os.path.join(self.root, "sequences", sequence, 'voxels', frame_id + ".occluded") 
        occluded = SemanticKittiIO._read_occluded_SemKITTI(occluded_path)
        occluded = np.moveaxis(occluded.reshape([self.grid_dimensions[0],
                                                    self.grid_dimensions[2],
                                                    self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1]) 
        data['occluded'] = occluded
    
        # rgb img
        rgb_path = os.path.join(self.root, "sequences", sequence, "image_2", frame_id + ".png")
        img = self._read_rgb(rgb_path)
        data['img'] = self.transforms_rgb (img)
        
        # depth map
        depth_path = os.path.join(self.root, "sequences", sequence, "dense_depth", frame_id + ".png")
        depth = self._read_depth(depth_path)
        depth_tensor = torch.tensor(depth)
        depth_tensor = depth_tensor.reshape((1,) + depth_tensor.shape)
        data["depth"] = depth_tensor        
        
        # target 1-1 and invalid 1-1
        target_1_1_path = os.path.join(self.root, "sequences", sequence, 'voxels', frame_id + ".label")  
        invalid_1_1_path =  os.path.join(self.root, "sequences",sequence, 'voxels', frame_id + ".invalid")        
        data['target_1_1'],data['invalid_1_1'] = self.get_label_at_scale('1_1', target_1_1_path,invalid_1_1_path)
        fov_target_1_1 = np.moveaxis(data['target_1_1'].copy(),[0, 1, 2], [0, 2, 1]).reshape(-1)
        fov_target_1_1[data["fov_mask_1"]==False] = 0
        
        # occluded_mask = data['occluded'].reshape(-1)
        # fov_target_1_1[occluded_mask==1] = 0
        
        
        data['fov_target_1_1'] = np.moveaxis(fov_target_1_1.reshape(self.grid_dimensions[0],
                                                                    self.grid_dimensions[2],
                                                                    self.grid_dimensions[1]),[0, 1, 2], [0, 2, 1])
        
        
        # target 1-2 and invalid 1-2
        target_1_2_path = os.path.join(self.root, "sequences", sequence, 'voxels', frame_id + ".label_1_2")  
        invalid_1_2_path =  os.path.join(self.root, "sequences",sequence, 'voxels', frame_id + ".invalid_1_2")        
        data['target_1_2'], data['invalid_1_2'] = self.get_label_at_scale('1_2', target_1_2_path,invalid_1_2_path)
        fov_target_1_2 = np.moveaxis(data['target_1_2'].copy(),[0, 1, 2], [0, 2, 1]).reshape(-1) 
        fov_target_1_2[data["fov_mask_2"]==False] = 0
        data['fov_target_1_2'] = np.moveaxis(fov_target_1_2.reshape(self.grid_dimensions[0]//2,
                                                                    self.grid_dimensions[2]//2,
                                                                    self.grid_dimensions[1]//2),[0, 1, 2], [0, 2, 1])
        
        # target 1-4 and invalid 1-4
        target_1_4_path = os.path.join(self.root, "sequences", sequence, 'voxels', frame_id + ".label_1_4")  
        invalid_1_4_path =  os.path.join(self.root, "sequences",sequence, 'voxels', frame_id + ".invalid_1_4")        
        data['target_1_4'],data['invalid_1_4']  = self.get_label_at_scale('1_4', target_1_4_path,invalid_1_4_path)
        fov_target_1_4 = np.moveaxis(data['target_1_4'].copy(),[0, 1, 2], [0, 2, 1]).reshape(-1)    
        fov_target_1_4[data["fov_mask_4"]==False] = 0
        data['fov_target_1_4'] =  np.moveaxis(fov_target_1_4.reshape(self.grid_dimensions[0]//4,
                                                                        self.grid_dimensions[2]//4,
                                                                        self.grid_dimensions[1]//4),[0, 1, 2], [0, 2, 1])
        
        # target 1-8 and invalid 1-8
        target_1_8_path = os.path.join(self.root, "sequences", sequence, 'voxels', frame_id + ".label_1_8")  
        invalid_1_8_path =  os.path.join(self.root, "sequences",sequence, 'voxels', frame_id + ".invalid_1_8")        
        data['target_1_8'],data['invalid_1_8'] = self.get_label_at_scale('1_8', target_1_8_path,invalid_1_8_path)   
        fov_target_1_8 = np.moveaxis(data['target_1_8'].copy(),[0, 1, 2], [0, 2, 1]).reshape(-1)     
        fov_target_1_8[data["fov_mask_8"]==False] = 0
        data['fov_target_1_8'] =  np.moveaxis(fov_target_1_8.reshape(self.grid_dimensions[0]//8,
                                                                        self.grid_dimensions[2]//8,
                                                                        self.grid_dimensions[1]//8),[0, 1, 2], [0, 2, 1])

        
        if self.istest:
            nonempty_1_1 = self.get_nonempty('1_1',invalid_1_1_path)
            nonempty_1_2 = self.get_nonempty('1_2',invalid_1_2_path)
            nonempty_1_4 = self.get_nonempty('1_4',invalid_1_4_path)
            nonempty_1_8 = self.get_nonempty('1_8',invalid_1_8_path)
            data['nonempty_1_1'] = nonempty_1_1
            data['nonempty_1_2'] = nonempty_1_2
            data['nonempty_1_4'] = nonempty_1_4
            data['nonempty_1_8'] = nonempty_1_8
                
            return data, sequence, frame_id
        # ---------------------------------------------------------------------------
        # Processing repackaged data provided by DDRNet
        # ---------------------------------------------------------------------------

        return  data, sequence, frame_id  
        

    def __len__(self):
        return len(self.filepaths)

    def get_filelist(self, root, sequences):
        if root is None:
            raise Exception("Oops! 'root' is None, please set the right file path.")
        filepaths = []
        
        for sequence in sequences:
            calib = self.read_calib(os.path.join(root, "sequences", sequence, "calib.txt"))
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(root, "sequences", sequence, "voxels", "*.bin")
            for voxel_path in glob.glob(glob_path):
                filepaths.append({"sequence": sequence,
                                   "P": P,
                                   "T_velo_2_cam": T_velo_2_cam,
                                   "proj_matrix": proj_matrix,
                                   "voxel_path": voxel_path,})        
        

        if len(filepaths) == 0:
            raise Exception("Oops!  That was no valid data in '{}'.".format(root))

        return filepaths
    
    def get_packaged_filelist(self, root, sequences):
        if root is None:
            raise Exception("Oops! 'root' is None, please set the right file path.")
        filepaths = []
        
        for sequence in sequences:
            glob_path = os.path.join(root, "sequences", sequence, "*.npz")
            for path in glob.glob(glob_path):
                filepaths.append({"sequence": sequence,
                                  "path": path,})  
        if len(filepaths) == 0:
            raise Exception("Oops!  That was no valid data in '{}'.".format(root))  
        return filepaths              
                
    def get_label_at_scale(self, scale, label_filename, invalid_filename):

        scale_divide = int(scale[-1])
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_filename)
        LABEL = SemanticKittiIO._read_label_SemKITTI(label_filename)
        if scale == '1_1': 
            LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
        LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
        LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
                                        int(self.grid_dimensions[2] / scale_divide),
                                        int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])
        INVALID = np.moveaxis(INVALID.reshape([int(self.grid_dimensions[0] / scale_divide),
                              int(self.grid_dimensions[2] / scale_divide),
                              int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])
        
        return LABEL, INVALID
    
   
    def get_nonempty(self, scale, invalid_filename):
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_filename)
        scale_divide = int(scale[-1])
        INVALID = np.moveaxis(INVALID.reshape([int(self.grid_dimensions[0] / scale_divide),
                                              int(self.grid_dimensions[2] / scale_divide),
                                              int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])        
       
        ONE = np.ones_like(INVALID)
        ZERO = np.zeros_like(INVALID)
        nonempty = np.where(INVALID<0.1, ONE, ZERO)
            
        return nonempty
    def get_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(self.dataset_config['learning_map'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut
        
    @staticmethod
    def _read_depth(depth_filename):
        r"""Read a depth image with size H x W
        and save the depth values (in millimeters) into a 2d numpy array.
        The depth image file is assumed to be in 16-bit PNG format, depth in millimeters.
        """
        # depth = misc.imread(depth_filename) / 8000.0  # numpy.float64
        # depth = imageio.imread(depth_filename) / 8000.0  # numpy.float64
        # assert depth.shape == (img_h, img_w), 'incorrect default size'
        
        depth = cv2.imread(depth_filename,-1)
        depth=cv2.split(depth)[0]
        # depth = cv2.flip(depth, 1)
        depth = depth/8000
        # depth = np.asarray(depth)
        
        
        depth = depth[:370, :1220]
        return depth

    @staticmethod
    def _read_rgb(rgb_filename):  # 0.01s
        r"""Read a RGB image with size H x W
        """
        img = Image.open(rgb_filename).convert("RGB")
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=True) / 255.0
     
        img = img[:370, :1220, :]  # crop image
        
        
        # rgb = misc.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        # rgb = imageio.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        # rgb = np.rollaxis(rgb, 2, 0)  # (H, W, 3)-->(3, H, W)
        return img

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out
    

    @staticmethod
    def _get_xyz(size):
        """x 水平 y高低  z深度"""
        _x = np.zeros(size, dtype=np.int32)
        _y = np.zeros(size, dtype=np.int32)
        _z = np.zeros(size, dtype=np.int32)

        for i_h in range(size[0]):  # x, y, z
            _x[i_h, :, :] = i_h                 # x, left-right flip
        for i_w in range(size[1]):
            _y[:, i_w, :] = i_w                 # y, up-down flip
        for i_d in range(size[2]):
            _z[:, :, i_d] = i_d                 # z, front-back flip
        return _x, _y, _z



if __name__ == '__main__':
    # ---- Data loader
    # data_dir = '/data/elon/NYU_SSC/NYUCADtrain_npz'
    data_dir = '/data/elon/MMSCNet/semantic_kitti'
    out_dir = '/data/elon/MMSCNet/train' 
    from PIL import Image
    import matplotlib.pyplot as plt
    import imageio
    import numpy as np
    import cv2

    # ------------------------------------------------ #
    data_set = SemanticKittiDataset(data_dir,'train',istest=True)
    print("len = {}".format(len(data_set)))
    num_data = len(data_set)
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

    for i in range(num_data):
        print("processing {}|{} ...".format(num_data, i))
        data, sequence, frame_id = data_set[i]
        
        zip_data_path = os.path.join(out_dir, sequence)
        if not os.path.isdir(zip_data_path):
            os.makedirs(zip_data_path)
        np.savez(zip_data_path +'/'+frame_id,
                 sequence=sequence,
                 frame_id=frame_id,
                 rgb=data['img'],
                 depth=data['depth'],
                 occupancy=data['occupancy'],
                 occluded=data['occluded'],
                 invalid_1_1=data['invalid_1_1'],   
                 invalid_1_2=data['invalid_1_2'],   
                 invalid_1_4=data['invalid_1_4'],   
                 invalid_1_8=data['invalid_1_8'],                 
                 target_1_1=data['target_1_1'],
                 target_1_2=data['target_1_2'],
                 target_1_4=data['target_1_4'],
                 target_1_8=data['target_1_8'],
                 fov_target_1_1=data['fov_target_1_1'],
                 fov_target_1_2=data['fov_target_1_2'],
                 fov_target_1_4=data['fov_target_1_4'],
                 fov_target_1_8=data['fov_target_1_8'],
                 projected_pix_1=data['projected_pix_1'],
                 projected_pix_2=data['projected_pix_2'],
                 projected_pix_4=data['projected_pix_4'],
                 projected_pix_8=data['projected_pix_8'],
                 fov_mask_1=data['fov_mask_1'],
                 fov_mask_2=data['fov_mask_2'],
                 fov_mask_4=data['fov_mask_4'],
                 fov_mask_8=data['fov_mask_8'],
                 nonempty_1_1=data['nonempty_1_1'],
                 nonempty_1_2=data['nonempty_1_2'],
                 nonempty_1_4=data['nonempty_1_4'],
                 nonempty_1_8=data['nonempty_1_8'])
        