import torch
import torch.nn as nn


class FLoSP(nn.Module):
    def __init__(self, scene_size, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, fov_mask):
        c, h, w = x2d.shape

        src = x2d.view(c, -1)
        # print("src = {}".format(src.shape))
        zeros_vec = torch.zeros(c, 1).type_as(src)
        # print("zeros_vec = {}".format(zeros_vec.shape))
        src = torch.cat([src, zeros_vec], 1)
        # print("src_cat = {}".format(src.shape))
        
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices[~fov_mask] = h * w
        img_indices = img_indices.expand(c, -1).long()  # c, HWD
        src_feature = torch.gather(src, 1, img_indices)

       
        
        x3d = src_feature.reshape(
            c,
            self.scene_size[0] // self.project_scale,
            self.scene_size[1] // self.project_scale,
            self.scene_size[2] // self.project_scale,
        )
        x3d = x3d.permute(0, 1, 3, 2)

        return x3d