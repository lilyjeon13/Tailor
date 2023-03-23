# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse

import numpy as np
import torch

import smplx

import sys
sys.path.append('../utils')
from utils.utils import Utils

def smplpose2smplxpose(smplpose):
    '''
    INPUT
        smplpose: [N, 24 * 3]
    OUTPUT
        smplxpose: [N, 21 * 3]
    '''
    smplpose = smplpose.reshape([-1, 24, 3])
    smplxpose = smplpose[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
    smplxpose = smplxpose.reshape([-1, 21 * 3])
    return smplxpose

def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         num_expression_coeffs=10):

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    print(model)

    betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    expression = torch.randn(
        [1, model.num_expression_coeffs], dtype=torch.float32)

    #body_pose = torch.randn(
    #    [1, model.NUM_BODY_JOINTS, 3], dtype=torch.float32) * 0.0
    #body_pose[:, 1] += 1
    #body_pose = body_pose.view([1, -1])
    body_pose = torch.FloatTensor([[-0.09681905061006546, -0.03425717353820801, 0.030272137373685837, 0.18732109665870667, 0.15736013650894165, 0.0922677293419838, 0.24782311916351318, -0.3062630891799927, -0.14813724160194397, 0.08571051806211472, 0.004716965835541487, 0.14138393104076385, -0.1410536766052246, -0.04115522652864456, -0.061735574156045914, -0.12601728737354279, -0.20469997823238373, 0.05190874636173248, -0.3144761025905609, 0.05176307633519173, -0.08825551718473434, 0.13344736397266388, 0.2245791107416153, -0.21885357797145844, 0.18842996656894684, -0.13706348836421967, 0.1572127342224121, 0.3086540102958679, 0.009847837500274181, -0.10058707743883133, -0.08821946382522583, 0.017798731103539467, 0.2307359278202057, -0.24255089461803436, -0.30764061212539673, -0.19544371962547302, 0.3383592665195465, 0.15434494614601135, 0.1321900635957718, -0.4681617319583893, 0.22779157757759094, 0.025413736701011658, -0.1858500838279724, 0.06513457745313644, 0.04978000000119209, 0.07719670236110687, -0.1145344004034996, -0.47248613834381104, 0.24021629989147186, -0.736968994140625, -1.239113211631775, 0.45409613847732544, 0.10593412816524506, 1.233400583267212, 0.447902649641037, -1.9297162294387817, 0.7999812364578247, 0.6600711941719055, 1.0941388607025146, 0.062452852725982666, 0.027025967836380005, -0.22080282866954803, 1.129733681678772, 0.2533450126647949, 0.20894865691661835, -0.724321186542511, -1.2145293951034546, 0.0555441677570343, -0.956276535987854, -0.8460237979888916, 0.21015393733978271, 0.919246256351471]])
    body_pose = smplpose2smplxpose(body_pose)

    output = model(betas=betas, expression=expression, body_pose=body_pose,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('pose shape =', body_pose.shape)
    print('Joints shape =', joints.shape)

    obj_dict = Utils.OBJ.load_obj('./samples/smplx.obj')
    obj_dict['verts'] = vertices
    obj_dict['normals'] = []
    obj_dict['fNormals'] = []
    Utils.OBJ.save_obj(obj_dict, './samples/smplx_example.obj')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='male',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=300, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')


    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    gender = args.gender
    ext = args.ext
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs)
