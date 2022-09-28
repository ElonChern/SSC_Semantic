#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
python script to evaluate the SSC model
---
Jie Li
jieli_cn@163.com
Nanjing University of Science and Technology
Aug 25, 2019
"""


import os
import torch
import argparse
import datetime
import numpy as np
from dataloaders import make_data_loader
from models import make_model
# from main import validate_on_dataset_stsdf
import config
import sscMetrics
from torch.autograd import Variable
from tqdm import tqdm
from dataloaders.semantic_kitti import SemanticKittiDataset

parser = argparse.ArgumentParser(description='PyTorch SSC Training')
parser.add_argument('--dataset', type=str, default='semantic_kitti', choices=['nyu', 'nyucad', 'debug'],
                    help='dataset name (default: nyu)')
parser.add_argument('--model', type=str, default='ddrnet', choices=['ddrnet', 'aicnet', 'grfnet', 'palnet', 'lwddrnet'],
                    help='model name (default: palnet)')
parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--resume', type=str, default='/home/elon/SSC-Semantic/cpBest_SSC_debug.pth.tar',metavar='PATH', help='path to latest checkpoint (default: none)')


global args
args = parser.parse_args()

def validate_on_dataset_stsdf(model, date_loader, save_ply=True):
    """
    Evaluate on validation set.
        model: network with parameters loaded
        date_loader: TEST mode
    """
    model.eval()  # switch to evaluate mode.
    val_acc, val_p, val_r, val_iou = 0.0, 0.0, 0.0, 0.0
    _C = 20
    val_cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
    val_iou_ssc = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
    count = 0
    with torch.no_grad():
        # ---- STSDF  depth, input, target, position, _
        save_output = {}
        for step, (data,sequence,filename ) in tqdm(enumerate(date_loader), desc='Validating', unit='frame'):
            var_x_depth = Variable(data['depth'].float()).cuda()
            y_true = data['fov_target_1_4'].long()
            nonempty = data['nonempty_1_4']

            if args.model == 'palnet':
                pass
                # var_x_volume = Variable(volume.float()).cuda()
                # y_pred = model(x_depth=var_x_depth, x_tsdf=var_x_volume, p=position)
            else:
                var_x_rgb = Variable(data['img'].float()).cuda()
                y_pred = model(x_depth=var_x_depth, x_rgb=var_x_rgb, data=data)  # y_pred.size(): (bs, C, W, H, D)

            y_pred = y_pred.cpu().data.numpy()  # CUDA to CPU, Variable to numpy
            y_true = y_true.cpu().data.numpy()  # torch tensor to numpy
            nonempty = nonempty.numpy()

            p, r, iou, acc, iou_sum, cnt_class = validate_on_batch(y_pred, y_true, nonempty)
            count += 1
            val_acc += acc
            val_p += p
            val_r += r
            val_iou += iou
            val_iou_ssc = np.add(val_iou_ssc, iou_sum)
            val_cnt_class = np.add(val_cnt_class, cnt_class)
            # print('acc_w, acc, p, r, iou', acc_w, acc, p, r, iou)
            
            if save_ply:
                output_preddir = '/home/elon/SSC-Semantic/output/pred'
                output_truedir = '/home/elon/SSC-Semantic/output/true'
                if not os.path.isdir(output_preddir):
                    os.makedirs(output_preddir)
                if not os.path.isdir(output_truedir):
                    os.makedirs(output_truedir)                  
                
                predict = np.argmax(y_pred, axis=1)
                # ---- check empty
                if nonempty is not None:
                    predict[nonempty == 0] = 0     # 0 empty
                _bs = predict.shape[0]      
                for idx in range(_bs):
                    pred_filename = os.path.join(output_preddir,  filename[idx] + ".ply")
                    true_filename = os.path.join(output_truedir,  filename[idx] + ".ply")
                    b_true = y_true[idx, :]  # GT
                    b_pred = predict[idx, :]
                    
                    SemanticKittiDataset.labeled_voxel2ply(b_true.T.astype(np.int32)  , true_filename)
                    SemanticKittiDataset.labeled_voxel2ply(b_pred.T.astype(np.int32)  , pred_filename)

            
    val_acc = val_acc / count
    val_p = val_p / count
    val_r = val_r / count
    val_iou = val_iou / count
    val_iou_ssc, val_iou_ssc_mean = sscMetrics.get_iou(val_iou_ssc, val_cnt_class)
    return val_p, val_r, val_iou, val_acc, val_iou_ssc, val_iou_ssc_mean


def validate_on_batch(predict, target, nonempty=None):  # CPU
    """
        predict: (bs, channels, D, H, W)
        target:  (bs, channels, D, H, W)
    """
    # TODO: validation will increase the usage of GPU memory!!!
    y_pred = predict
    y_true = target
    p, r, iou = sscMetrics.get_score_completion(y_pred, y_true, nonempty)
    #acc, iou_sum, cnt_class = sscMetrics.get_score_semantic_and_completion(y_pred, y_true, stsdf)
    acc, iou_sum, cnt_class, tp_sum, fp_sum, fn_sum = sscMetrics.get_score_semantic_and_completion(y_pred, y_true, nonempty)
    # iou = np.divide(iou_sum, cnt_class)
    return p, r, iou, acc, iou_sum, cnt_class

def main():
    # ---- Check CUDA
    if torch.cuda.is_available():
        print("Great, You have {} CUDA device!".format(torch.cuda.device_count()))
    else:
        print("Sorry, You DO NOT have a CUDA device!")

    train_time_start = datetime.datetime.now()
    test()
    print('Training finished in: {}'.format(datetime.datetime.now() - train_time_start))


def test():
    # ---- create model ---------- ---------- ---------- ---------- ----------#
    net = make_model(args.model, num_classes=20).cuda()
    net = torch.nn.DataParallel(net)  # Multi-GPU

    # ---- load pretrained model --------- ---------- ----------#
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        cp_states = torch.load(args.resume)
        net.load_state_dict(cp_states['state_dict'], strict=True)
    else:
        raise Exception("=> NO checkpoint found at '{}'".format(args.resume))

    # ---- Data loader
    train_loader, val_loader = make_data_loader(args)

    torch.cuda.empty_cache()

    # ---- Evaluation
    v_prec, v_recall, v_iou, v_acc, v_ssc_iou, v_mean_iou = validate_on_dataset_stsdf(net, val_loader)
    print('Validate with TSDF:, p {:.1f}, r {:.1f}, IoU {:.1f}'.format(v_prec*100.0, v_recall*100.0, v_iou*100.0))
    print('pixel-acc {:.4f}, mean IoU {:.1f}, SSC IoU:{}'.format(v_acc*100.0, v_mean_iou*100.0, v_ssc_iou*100.0))
    
if __name__ == '__main__':
    main()


