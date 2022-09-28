import os
import yaml
import numpy as np
import sys 
import torch
import open3d as o3d
import open3d
sys.path.append('/home/elon/SSC-Semantic/')
from config import colorMap
sys.path.append('/home/elon/Projects/PMF-master/')


class SemanticKitti(torch.utils.data.Dataset):
    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 config_path,  # directory of config file
                ):
        self.root = root
        self.sequences = sequences
        self.sequences.sort()  # sort seq id


        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, "r"))
        else:
            raise ValueError("config file not found: {}".format(config_path))

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        self.pointcloud_files = []


        for seq in self.sequences:
            # format seq id
            seq = "{0:02d}".format(int(seq))
            print("parsing seq {}...".format(seq))

            # get file list from path
            pointcloud_path = os.path.join(self.root, seq, "velodyne")
            pointcloud_files = [os.path.join(pointcloud_path, f) for f in os.listdir(
                pointcloud_path) if ".bin" in f]

            self.pointcloud_files.extend(pointcloud_files)


        # sort for correspondance 
        self.pointcloud_files.sort()

        print("Using {} pointclouds from sequences {}".format(
            len(self.pointcloud_files), self.sequences))

   
    def readPCD(self,path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pcd
    
    def voxel_filter(self, point_cloud, leaf_size):
        filtered_points = []
        print(point_cloud.shape)
        # 计算边界点
        x_min, y_min, z_min = np.amin(point_cloud[:,0:3], axis=0)  # 计算x y z 三个维度的最值
        x_max, y_max, z_max = np.amax(point_cloud[:,0:3], axis=0)
        # 计算 voxel grid维度
        Dx = (x_max - x_min) // leaf_size + 1
        Dy = (y_max - y_min) // leaf_size + 1
        Dz = (z_max - z_min) // leaf_size + 1
        print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))
        # 计算每个点的voxel索引
        h = list()  # h 为保存索引的列表
        for i in range(len(point_cloud)):
            hx = (point_cloud[i][0] - x_min) // leaf_size
            hy = (point_cloud[i][1] - y_min) // leaf_size
            hz = (point_cloud[i][2] - z_min) // leaf_size
            h.append(hx + hy * Dx + hz * Dx * Dy)
        h = np.array(h)
        # 筛选点
        h_indice = np.argsort(h)  # 返回h里面的元素按从小到大排序的索引
        h_sorted = h[h_indice]
        begin = 0
        for i in range(len(h_sorted) - 1):  # 0~9999
            if h_sorted[i] == h_sorted[i + 1]:
                continue
            else:
                point_idx = h_indice[begin: i + 1]
                xyz = np.mean(point_cloud[point_idx][:,:3], axis=0)
                intensity = np.mean(point_cloud[point_idx][:,3:4], axis=0)
                filtered_points.append(np.insert(xyz,3,intensity))
               
                begin = i+1
        # 把点云格式改成array，并对外返回
        filtered_points = np.array(filtered_points, dtype=np.float64)
        return filtered_points    
    
    def __getitem__(self, index): 
        filepath = self.pointcloud_files[index]
        pcd = self.readPCD(filepath)
        filtered_cloud = self.voxel_filter(pcd, 0.2)
        
        return pcd, filtered_cloud


    def __len__(self):
        return len(self.pointcloud_files)
    
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

def create_output(vertices, filename):
    # colors = colors.reshape(-1, 3)
    vertices = vertices.reshape(-1, 4)
    np.savetxt(filename, vertices, fmt='%f %f %f %f')     # 必须先写入，然后利用write()在头部插入ply header

                    
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
            property float intensity
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)   
        
         
if __name__ == '__main__':
   
    data_root = "/data/elon/semantic_kitti/sequences"
    data_config_path = "/home/elon/SSC-Semantic/dataloaders/semantic-kitti.yaml"
    dataset = SemanticKitti(root=data_root,
                            sequences=[0,1,2], 
                            config_path=data_config_path) 
    
    pcd, filtered_cloud = dataset[100]
    print(type(filtered_cloud))
    create_output(filtered_cloud, '1.ply')
    
  
    
    point_cloud = o3d.geometry.PointCloud()
    
    point_cloud.points = o3d.utility.Vector3dVector(filtered_cloud[:,0:3])
    
    # 显示滤波后的点云
    # o3d.visualization.draw_geometries([point_cloud])
    # o3d.io.write_point_cloud("copy_of_fragment.pcd",point_cloud)
    
        
    num_point = filtered_cloud.shape[0]
    voxel = np.zeros((1,256,256,32),dtype=np.int)
    for i in range(num_point):
        condition = filtered_cloud[i,0]<51.2 and filtered_cloud[i,0]>0 and\
                    filtered_cloud[i,1]<25.6 and  filtered_cloud[i,1]>-25.6 and\
                    filtered_cloud[i,2]<4.4 and  filtered_cloud[i,2]>-2
        if condition == True:
            x_idx = int((filtered_cloud[i,0]-0)/0.2) 
            y_idx = int((filtered_cloud[i,1]+25.6)/0.2) 
            z_idx = int((filtered_cloud[i,2]+2)/0.2) 
            
            voxel[0:1,x_idx,y_idx,z_idx] = 1
            voxel[1:2,x_idx,y_idx,z_idx] = filtered_cloud[i,3]
    
    
    SemanticKitti.labeled_voxel2ply(voxel[0,:,:,:]  , 'occupancy_filename.ply')



