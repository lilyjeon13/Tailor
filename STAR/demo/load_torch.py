# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman

from pytorch.star import STAR
star = STAR(gender='male')
import torch
import numpy as np 
from torch.autograd import Variable

unit_degree = np.pi * 0.5 # 90도
trivial_degree = unit_degree * 0.03

# max, min degree
pose_range = np.zeros([24, 3, 2]) # [#joint, #xyz, min & max]
# global
pose_range[0, :, :] = 0
# left thigh
pose_range[1, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[1, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[1, 2] = np.array([-trivial_degree, trivial_degree])
# right thigh
pose_range[2, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[2, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[2, 2] = np.array([-trivial_degree, trivial_degree])
# belly
pose_range[3, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[3, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[3, 2] = np.array([-trivial_degree, trivial_degree])
# left knee
pose_range[4, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[4, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[4, 2] = np.array([-trivial_degree, trivial_degree])
# right knee
pose_range[5, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[5, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[5, 2] = np.array([-trivial_degree, trivial_degree])
# upper belly
pose_range[6, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[6, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[6, 2] = np.array([-trivial_degree, trivial_degree])
# left ankle
pose_range[7, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[7, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[7, 2] = np.array([-trivial_degree, trivial_degree])
# right ankle
pose_range[8, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[8, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[8, 2] = np.array([-trivial_degree, trivial_degree])
# soloar plexus
pose_range[9, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[9, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[9, 2] = np.array([-trivial_degree, trivial_degree])
# left toe
pose_range[10, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[10, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[10, 2] = np.array([-trivial_degree, trivial_degree])
# right toe
pose_range[11, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[11, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[11, 2] = np.array([-trivial_degree, trivial_degree])
# neck
pose_range[12, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[12, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[12, 2] = np.array([-trivial_degree, trivial_degree])
# left chest
pose_range[13, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[13, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[13, 2] = np.array([-trivial_degree, trivial_degree])
# right chest
pose_range[14, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[14, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[14, 2] = np.array([-trivial_degree, trivial_degree])
# upper neck
pose_range[15, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[15, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[15, 2] = np.array([-trivial_degree, trivial_degree])
# left shoulder
pose_range[16, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[16, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[16, 2] = np.array([-trivial_degree, trivial_degree])
# right shoulder
pose_range[17, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[17, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[17, 2] = np.array([-trivial_degree, trivial_degree])
# left elbow
pose_range[18, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[18, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[18, 2] = np.array([-trivial_degree, trivial_degree])
# right elbow
pose_range[19, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[19, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[19, 2] = np.array([-trivial_degree, trivial_degree])
# left wrist
pose_range[20, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[20, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[20, 2] = np.array([-trivial_degree, trivial_degree])
# right wrist
pose_range[21, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[21, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[21, 2] = np.array([-trivial_degree, trivial_degree])
# left hand
pose_range[22, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[22, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[22, 2] = np.array([-trivial_degree, trivial_degree])
# right hand
pose_range[23, 0] = np.array([-trivial_degree, trivial_degree])
pose_range[23, 1] = np.array([-trivial_degree, trivial_degree])
pose_range[23, 2] = np.array([-trivial_degree, trivial_degree])

total_n = 20000
batch_size = 100
iter_ = total_n//batch_size

total_vertices = [] # [n, 6890, 3]
total_param = [] # [n, 85]
total_size = [] # [n, 3], height, width, thickness

for it in range(iter_):
    poses = np.zeros((batch_size,72))
    poses = np.array([[2.746571, 0.115941, 1.569221, -0.060806, -0.084250, -0.064932, -0.089765, -0.111454, 0.019431, -0.073432, 0.016814, 0.006608, 0.277896, -0.264463, -0.086147, 0.540536, -0.001556, 0.055530, 0.013326, -0.004305, -0.050642, -0.092524, 0.143655, 0.039586, 0.477736, -0.385644, 0.464008, 0.112884, -0.052172, -0.010707, -0.322734, 0.136646, 0.140619, -0.319016, 0.316203, -0.485310, 0.051791, -0.156329, 0.021439, 0.183554, 0.312552, -0.115679, 0.002789, -0.092230, 0.013133, 0.135542, -0.088762, 0.039004, 0.182390, -0.056961, -1.118093, 0.292830, 0.193644, 1.111046, 0.495261, -1.173295, 0.191734, 0.495906, 1.211908, -0.115371, -0.233130, -0.076959, -0.001271, 0.102211, 0.108801, 0.117276, -0.188209, -0.102066, -0.150867, -0.097702, 0.083491, 0.164768, ]])
    poses = poses.repeat(batch_size, 0)
    print(poses.shape)
    
    pose_range = pose_range.reshape([72, 2])
    poses[:, 0:3] = 0
    poses[:, 16*3] += np.pi / 6
    poses[:, 16*3 + 1] -= np.pi / 6
    poses[:, 16*3 + 2] += np.pi / 6
    poses[:, 18*3 + 1] -= np.pi / 16
    important_joints = [13,16,18,20,22, 9, 6, 3]
    important_joints_xyz = np.concatenate([[i * 3, i * 3 + 1, i * 3 + 2] for i in important_joints])
    for joint in range(72):
      if joint in important_joints_xyz:
        continue
      poses[:, joint] = np.random.random([batch_size]) * (pose_range[joint, 1] - pose_range[joint, 0]) + pose_range[joint, 0]
    '''
    # left side
    poses[:, 39] = np.pi/5 + np.random.normal(0, 1, [batch_size]) * np.pi / 30
    poses[:, 40] = 0 + np.random.normal(0, 1, [batch_size]) * np.pi / 30 #np.pi/32
    poses[:, 41] = -np.pi/4 + np.random.normal(0, 1, [batch_size]) * np.pi / 30
    poses[:, 48] = np.pi/5 + np.random.normal(0, 1, [batch_size]) * np.pi / 30
    poses[:, 55] = - np.pi/2 + np.random.normal(0, 1, [batch_size]) * np.pi / 30
    poses[:, 61] = - np.pi/8 + np.random.normal(0, 1, [batch_size]) * np.pi / 30
    poses[:, 62] = np.pi/8 + np.random.normal(0, 1, [batch_size]) * np.pi / 30
    
    poses[:, 39] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 40] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 41] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 48] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 49] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 50] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 54] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 55] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 61] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    poses[:, 62] += (np.random.random(batch_size) * 2 - 1) * np.pi / 10
    '''
    # right side
    for i in [39,48,54, 60, 66]:
        poses[:, i+3] = poses[:, i]
        poses[:, i+4] = -poses[:, i+1]
        poses[:, i+5] = -poses[:, i+2]

    #poses += np.random.normal(0, 1, [batch_size, 72]) * np.pi / 50
    
    
    poses = torch.cuda.FloatTensor(poses)
    poses = Variable(poses,requires_grad=True)
    betas = torch.cuda.FloatTensor( np.random.normal(0, 1, [batch_size, 10]) )
    betas = Variable(betas,requires_grad=True)
    trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
    trans = Variable(trans,requires_grad=True)
    
    vertices = star(poses, betas,trans)
    vertices_np = vertices.cpu().detach().numpy() # [n, 6890, 3]
    
    height = vertices_np[:, :, 1].max(axis = -1, keepdims = True) - vertices_np[:, :, 1].min(axis = -1, keepdims = True)
    width = vertices_np[:, :, 0].max(axis = -1, keepdims = True) - vertices_np[:, :, 0].min(axis = -1, keepdims = True)
    thickness = vertices_np[:, :, 2].max(axis = -1, keepdims = True) - vertices_np[:, :, 2].min(axis = -1, keepdims = True)
    size = np.concatenate([height, width, thickness], axis = -1)
    
    total_vertices.append(vertices_np)
    total_param.append(np.concatenate([poses.cpu().detach().numpy(), betas.cpu().detach().numpy(), trans.cpu().detach().numpy()], axis = -1))
    total_size.append(size)
    
    print('\r%d/%d'%( (it+1) * batch_size, total_n), end='')
np.save('./sample.npy', np.concatenate(total_vertices))
np.save('./sample_param.npy', np.concatenate(total_param))
np.save('./sample_size.npy', np.concatenate(total_size))
print('\ndone')