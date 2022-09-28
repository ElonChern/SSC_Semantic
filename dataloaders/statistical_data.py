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
from torch_scatter import scatter_max
sys.path.append('/home/elon/SSC-Semantic/')
from config import colorMap
from torchvision import transforms
from dataloaders import io_data as SemanticKittiIO
from dataloaders.utils.helpers import vox2pix


class SemanticKittiDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase,subfix='npz',istest=False):
        # self.param = {'voxel_size': (256, 32, 256),
        #               'voxel_unit': 0.2,            # 0.02m, length of each grid == 20mm
        #               'cam_k': [[718.856, 0, 607.1928],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
        #                         [0, 718.856, 185.2157],  # cx = K(1,3); cy = K(2,3);
        #                         [0, 0, 1]],          # fx = K(1,1); fy = K(2,2);
        #               }
        self.scene_size = (51.2, 51.2, 6.4)
        self.subfix = subfix
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
       
        if subfix == 'npz':
            self.filepaths = self.get_packaged_filelist(self.root, self.sequences)
        else:
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
           
        
        if self.subfix == 'npz':
            data = {}
            filepath = self.filepaths[index]
            sequence = filepath["sequence"]
            path = filepath["path"]
            filename = os.path.basename(path)
            frame_id = os.path.splitext(filename)[0]
            package_data_path = os.path.join(self.root, "sequences", sequence, frame_id + ".npz")
            with np.load(package_data_path) as npz_file:
                data['img'] = npz_file['rgb']
                data['depth'] = npz_file['depth']
                data['occluded'] = npz_file['occluded']
                data['occupancy'] = npz_file['occupancy']
                
                data['target_1_1'] = npz_file['target_1_1']
                data['target_1_2'] = npz_file['target_1_2']
                data['target_1_4'] = npz_file['target_1_4']
                data['target_1_8'] = npz_file['target_1_8']
                
                data['fov_target_1_1'] = npz_file['fov_target_1_1']
                data['fov_target_1_2'] = npz_file['fov_target_1_2']
                data['fov_target_1_4'] = npz_file['fov_target_1_4']
                data['fov_target_1_8'] = npz_file['fov_target_1_8']
                
                data['fov_mask_1'] = npz_file['fov_mask_1']
                data['fov_mask_2'] = npz_file['fov_mask_2']
                data['fov_mask_4'] = npz_file['fov_mask_4']
                data['fov_mask_8'] = npz_file['fov_mask_8']
                
                data['projected_pix_1'] = npz_file['projected_pix_1']
                data['projected_pix_2'] = npz_file['projected_pix_2']
                data['projected_pix_4'] = npz_file['projected_pix_4']
                data['projected_pix_8'] = npz_file['projected_pix_8']
                
                data['invalid_1_1'] = npz_file['invalid_1_1']
                data['invalid_1_2'] = npz_file['invalid_1_2']
                data['invalid_1_4'] = npz_file['invalid_1_4']
                data['invalid_1_8'] = npz_file['invalid_1_8']
               
                if self.istest:
                    data['nonempty_1_1'] = npz_file['nonempty_1_1']
                    data['nonempty_1_2'] = npz_file['nonempty_1_2']
                    data['nonempty_1_4'] = npz_file['nonempty_1_4']
                    data['nonempty_1_8'] = npz_file['nonempty_1_8']
                    return data, sequence, frame_id  
                    
            return data, sequence, frame_id      
        else:       
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
                
            pix_x, pix_y = data["projected_pix_1"][:,0], data["projected_pix_1"][:,1]
            img_indices = pix_y * 1220 + pix_x
            img_indices[data["fov_mask_1"]==False] = 1220 * 370
            img_indices = img_indices.astype(np.long)           

            
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
            
            src = np.moveaxis(data['target_1_1'],[0, 1, 2], [0, 2, 1]).reshape(-1)
            out = torch.zeros([1220*370])
            src = torch.from_numpy(src)
            img_indices = torch.from_numpy(img_indices)
            proj_target,_ = scatter_max(src,img_indices,out=out)
            proj_target = proj_target.reshape(370,1220)
            proj_target = np.expand_dims(proj_target, axis=0)
            # print('projec ={}'.format(proj_target.shape))
            data['proj_target'] = proj_target
            
            
            fov_target_1_1 = np.moveaxis(data['target_1_1'].copy(),[0, 1, 2], [0, 2, 1]).reshape(-1)
            fov_target_1_1[data["fov_mask_1"]==False] = 0
            
            # occluded_mask = np.moveaxis(data['occluded'].copy(),[0, 1, 2], [0, 2, 1]).reshape(-1)
            # fov_target_1_1[occluded_mask==0] = 0
            
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
        depth = imageio.imread(depth_filename) / 8000.0  # numpy.float64
        # assert depth.shape == (img_h, img_w), 'incorrect default size'
        
        depth = cv2.imread(depth_filename,-1)
        depth=cv2.split(depth)[0]
        depth = cv2.flip(depth, 1)
        depth = depth/8000
        # depth = np.asarray(depth)
        
        
        depth = depth[:370, :1220]
        return depth

    @staticmethod
    def _read_rgb(rgb_filename):  # 0.01s
        r"""Read a RGB image with size H x W
        """
        img = Image.open(rgb_filename).convert("RGB")
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
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

    @classmethod
    def labeled_voxel2ply(cls, vox_labeled, ply_filename):  #
        """Save labeled voxels to disk in colored-point cloud format: x y z r g b, with '.ply' suffix
           vox_labeled.shape: (W, H, D)
        """  #
        
       
        if type(vox_labeled) is torch.Tensor:
            vox_labeled = vox_labeled.numpy()  
            vox_labeled = vox_labeled.astype(np.int32) 
             
        # ---- Check data type, numpy ndarray
        if type(vox_labeled) is not np.ndarray:
            raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
        # ---- Check data validation
        if np.amax(vox_labeled) == 0:
            print('Oops! All voxel is labeled empty.')
            return
        # ---- get size
        size = vox_labeled.shape
        
        # print('vox_labeled.shape:', vox_labeled.shape)
        # ---- Convert to list
        vox_labeled = vox_labeled.flatten()
        # ---- Get X Y Z
        _x, _y, _z = cls._get_xyz(size)
        _x = _x.flatten()
        _y = _y.flatten()
        _z = _z.flatten()
        # print('_x.shape', _x.shape)
        # ---- Get R G B
        vox_labeled[vox_labeled == 255] = 0  # empty
        # vox_labeled[vox_labeled == 255] = 12  # ignore
        _rgb = colorMap[vox_labeled[:]]
        # print('_rgb.shape:', _rgb.shape)
        # ---- Get X Y Z R G B
        xyz_rgb = zip(_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # python2.7
        xyz_rgb = list(xyz_rgb)  # python3
        # print('xyz_rgb.shape-1', xyz_rgb.shape)
        # xyz_rgb = zip(_z, _y, _x, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # 将X轴和Z轴交换，用于meshlab显示
        # ---- Get ply data without empty voxel

        xyz_rgb = np.array(xyz_rgb)
        # print('xyz_rgb.shape-1', xyz_rgb.shape)
        ply_data = xyz_rgb[np.where(vox_labeled > 0)]

        if len(ply_data) == 0:
            raise Exception("Oops!  That was no valid ply data.")
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % len(ply_data)
        # ---- Save ply data to disk
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
        del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head
        # print('Saved-->{}'.format(ply_filename))


if __name__ == '__main__':
    # ---- Data loader
    # data_dir = '/data/elon/NYU_SSC/NYUCADtrain_npz'
    data_dir = '/data/elon/MMSCNet/semantic_kitti_package'
    # data_dir = '/data/elon/MMSCNet/semantic_kitti'
    from PIL import Image
    import matplotlib.pyplot as plt
    import imageio
    import numpy as np
    import cv2

    # ------------------------------------------------ #
    data_set = SemanticKittiDataset(data_dir,'val',subfix='npz',istest=True)
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4)
      
    root = '/home/elon/SSC-Semantic/output'
    
    cal_label = {"unlabeled":0,
                 "car":0,
                 "bicycle":0,
                 "motorcycle":0,
                 "truck":0,
                 "other-vehicle":0,
                 "person":0,
                 "bicyclist":0,
                 "motorcyclist":0,
                 "road":0,
                 "parking":0,
                 "sidewalk":0,
                 "other-ground":0,
                 "building":0,
                 "fence":0,
                 "vegetation":0,
                 "trunk":0,
                 "terrain":0,
                 "pole":0,
                 "traffic-sign":0}
    
    cal_layer = {'00':cal_label.copy(),'01':cal_label.copy(),'02':cal_label.copy(),'03':cal_label.copy(),
                 '04':cal_label.copy(),'05':cal_label.copy(),'06':cal_label.copy(),'07':cal_label.copy(),
                 '08':cal_label.copy(),'09':cal_label.copy(),'10':cal_label.copy(),'11':cal_label.copy(),
                 '12':cal_label.copy(),'13':cal_label.copy(),'14':cal_label.copy(),'15':cal_label.copy(),
                 '16':cal_label.copy(),'17':cal_label.copy(),'18':cal_label.copy(),'19':cal_label.copy(),
                 '20':cal_label.copy(),'21':cal_label.copy(),'22':cal_label.copy(),'23':cal_label.copy(),
                 '24':cal_label.copy(),'25':cal_label.copy(),'26':cal_label.copy(),'27':cal_label.copy(),
                 '28':cal_label.copy(),'29':cal_label.copy(),'30':cal_label.copy(),'31':cal_label.copy(),}
    
    percent_layer = {'00':0,'01':0,'02':0,'03':0,
                    '04':0,'05':0,'06':0,'07':0,
                    '08':0,'09':0,'10':0,'11':0,
                    '12':0,'13':0,'14':0,'15':0,
                    '16':0,'17':0,'18':0,'19':0,
                    '20':0,'21':0,'22':0,'23':0,
                    '24':0,'25':0,'26':0,'27':0,
                    '28':0,'29':0,'30':0,'31':0,}    
    
    percent_label = {"unlabeled": percent_layer.copy(),"car":percent_layer.copy(),"bicycle":percent_layer.copy(),
                     "motorcycle":percent_layer.copy(),"truck":percent_layer.copy(),"other-vehicle":percent_layer.copy(),
                     "person":percent_layer.copy(),"bicyclist":percent_layer.copy(),"motorcyclist":percent_layer.copy(),
                     "road":percent_layer.copy(),"parking":percent_layer.copy(),"sidewalk":percent_layer.copy(),
                     "other-ground":percent_layer.copy(),"building":percent_layer.copy(),"fence":percent_layer.copy(),
                     "vegetation":percent_layer.copy(),"trunk":percent_layer.copy(),"terrain":percent_layer.copy(),
                     "pole":percent_layer.copy(),"traffic-sign":percent_layer.copy()}


    label = ["unlabeled","car","bicycle","motorcycle","truck",
                 "other-vehicle","person","bicyclist","motorcyclist",
                 "road","parking","sidewalk","other-ground","building",
                 "fence","vegetation","trunk","terrain","pole","traffic-sign"]
    
    layer = ['00','01','02','03','04','05','06',
            '07','08','09','10','11','12',
            '13','14','15','16','17','18',
            '19','20','21','22','23','24',
            '25','26','27','28','29','30','31',]
    
    label_map = {"unlabeled":0,"car":1,"bicycle":2,"motorcycle":3,"truck":4,
                 "other-vehicle":5,"person":6,"bicyclist":7,"motorcyclist":8,
                 "road":9,"parking":10,"sidewalk":11,"other-ground":12,"building":13,
                 "fence":14,"vegetation":15,"trunk":16,"terrain":17,"pole":18,"traffic-sign":19}
    
    class_total_num = {"unlabeled":0,
                        "car":0,
                        "bicycle":0,
                        "motorcycle":0,
                        "truck":0,
                        "other-vehicle":0,
                        "person":0,
                        "bicyclist":0,
                        "motorcyclist":0,
                        "road":0,
                        "parking":0,
                        "sidewalk":0,
                        "other-ground":0,
                        "building":0,
                        "fence":0,
                        "vegetation":0,
                        "trunk":0,
                        "terrain":0,
                        "pole":0,
                        "traffic-sign":0}
   
    def count_label(label,data):
        max = label + 1
        min = label -1
        ONE = torch.ones_like(data)
        ZERO = torch.zeros_like(data)
        CONDITION = (data<max) * (data>min)
        CAL_LABEL = torch.where(CONDITION, ONE, ZERO)
        num = torch.sum(CAL_LABEL)  
        return num   
        
        
               
    # for step, (data, sequence,filename) in enumerate(data_loader):
    #     print('step = {}|{} ... '.format(step,len(data_set)) )
       
    #     _bs = data['img'].shape[0]  
            
    #     for idx in range(_bs):
                          
    #         b_target_1_1 = data['target_1_1'][idx]  # GT
    #         b_target_1_2 = data['target_1_2'][idx]  # GT
    #         b_target_1_4 = data['target_1_4'][idx]  # GT
    #         b_target_1_8 = data['target_1_8'][idx]  # GT           
           
    #         for i in range(len(layer)):

    #             layer_target_1_1 = b_target_1_1[:,i,:] 
                
    #             for j in range(len(label)):
    #                 cal_layer[layer[i]][label[j]] = cal_layer[layer[i]][label[j]] + count_label(label_map[label[j]], layer_target_1_1)
                                   
    # np.save('cal_layer.npy',cal_layer)
    
    cal_layer = np.load('cal_layer.npy',allow_pickle = True).item()
    
    for i in range(len(label)):
        
        for j in range(len(layer)):
            percent_label[label[i]][layer[j]] = cal_layer[layer[j]][label[i]] + percent_label[label[i]][layer[j]]
            
    for i in range(len(label)):
        
        for j in range(len(layer)): 
            class_total_num[label[i]] = class_total_num[label[i]] +  percent_label[label[i]][layer[j]]       
    
    for i in range(len(label)):
        for j in range(len(layer)): 
            percent_label[label[i]][layer[j]] = (percent_label[label[i]][layer[j]] / class_total_num[label[i]])*100
            
    print('percent_label = {}'.format(percent_label))
                    
                    
               
             
            

            
