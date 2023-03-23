def log(message, prefix = '>>>', silent = False):
    if not silent:
        print(prefix, message)
def error(message, prefix = '!!!!!!'):
    print(prefix, message)

ROOT_NAME = 'Metown'
import os
import re
cur = os.getcwd() # get current path
ROOT_DIR = cur[:list(re.finditer(ROOT_NAME, cur))[-1].start()] + ROOT_NAME # get root path
log('UTILS: root path: %s'%ROOT_DIR)
import sys
sys.path.append(ROOT_DIR)
STAR_DIR = os.path.join(ROOT_DIR, 'STAR')
sys.path.append(STAR_DIR); from pytorch.star import STAR as STAR_; sys.path.remove(STAR_DIR)
SMPLX_DIR = os.path.join(ROOT_DIR, 'smplx')
sys.path.append(SMPLX_DIR); import smplx; sys.path.remove(SMPLX_DIR)

import numpy as np
import torch
import pickle
import json
import time

class Utils:
    class PARSER:
        def get_parser():
            import argparse
            return argparse.ArgumentParser()
        def parsing(parser):
            return parser.parse_args()
    class IO:
        def write_file(file_path, contents):
            f = open(file_path, 'w')
            f.write(contents)
            f.close()
        def write_json(file_path, dict_):
            with open(file_path, 'w') as fp:
                json.dump(dict_, fp)
        def read_json(file_path):
            with open(file_path, "r") as fp:
                dict_ = json.load(fp)
            return dict_
    class STAR:
        def __init__(self, gender = 'male', num_betas = 10):
            self.star = STAR_(gender = gender, num_betas = num_betas)
            self.num_betas = num_betas
            log('Star initialized.')
        def forward(self, pose, shape, trans, disps = None):
            vertices = self.star(pose, shape, trans, disps)
            self.joint_locs = self.star.joint_locs
            return vertices
        def SMPLpose2MTHpose(self, pose):
            return pose
        def load_star(gender = 'male', num_betas = 10):
            star = STAR_(gender = gender, num_betas = num_betas)
            log('Star loaded completely.')
            return star
        def generate_random_star_param(N, seed = None, pose_rho = 0, pose_sig = 1, shape_rho = 0, shape_sig = 1, trans_rho = 0, trans_sig = 0, num_betas = 10):
            '''
            *** INPUT ***
                N: how many parames would be generated.
                seed: random seed (default: None)
                pose_rho: mean of pose
                pose_sig: standard deviation of pose (same way for shape and trans. trans sig is set to 0 in default, because trans is usually zero vector.)
            *** OUTPUT ***
                pose: [N, 72]
                shape: [N, 10]
                trans: [N, 3]
            '''
            if seed is not None:
                np.random.seed(seed)
            pose = np.random.normal(pose_rho, pose_sig, size=(N, 72))
            shape = np.random.normal(shape_rho, shape_sig, size=(N, num_betas))
            trans = np.random.normal(trans_rho, trans_sig, size=(N, 3))
            return pose, shape, trans
        def star_forward(star, pose, shape, trans, disps = None, to_numpy = True):
            '''
            *** INPUT ***
                star: star model
                pose: [N, 72]
                shape: [N, 10]
                trans: [N, 3]
                disps: [N, 6890, 3]
            *** OUTPUT ***
                vertices [N, 6890, 3]
            '''
            if type(pose) == np.ndarray:
                pose = torch.FloatTensor(pose)
            if type(shape) == np.ndarray:
                shape = torch.FloatTensor(shape)
            if type(trans) == np.ndarray:
                trans = torch.FloatTensor(trans)
            if disps is not None:
                if type(disps) == np.ndarray:
                    disps = torch.FloatTensor(disps)
                disps = disps.to('cuda')

            vertices = star(pose.to('cuda'), shape.to('cuda'), trans.to('cuda'), disps).to('cpu')
            if to_numpy:
                vertices = vertices.numpy()
            return vertices
        def save_star_obj(verts, save_path):
            '''
            *** INPUT ***
                verts: [6890, 3]
                save_path: path to save obj file
            *** OUTPUT ***
                None
            '''
            star_json_path = os.path.join(STAR_DIR, 'model/neutral_smpl_with_cocoplus_reg.txt')
            with open(star_json_path, 'r') as reader:
                faces = json.load(reader)['f']
            sample_star_path = os.path.join(STAR_DIR, 'sampleSTAR.obj') # needed for uv map
            if not save_path.endswith('.obj'):
                save_path += '.obj'
            textures = []
            #faces_verts = []
            faces_textures = []
            with open(sample_star_path, 'r') as fr:
                while True:
                    line = fr.readline()
                    if not line: break
                    words = line.split(' ')
                    if words[0] == 'f':
                        #faces_verts.append([int(words[1].split('/')[0]), int(words[2].split('/')[0]), int(words[3].split('/')[0])])
                        faces_textures.append([int(words[1].split('/')[1]), int(words[2].split('/')[1]), int(words[3].split('/')[1])])
                    elif words[0] == 'vt':
                        textures.append([float(words[1]), float(words[2])])
            fr.close()
            with open(save_path, 'w') as fp:
                for v in verts:
                    fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
                for t in textures:
                    fp.write( 'vt %f %f\n' % ( t[0], t[1]) )
                for i in range(len(faces)): # Faces are 1-based, not 0-based in obj files
                    fp.write( 'f %d/%d %d/%d %d/%d\n' %  (faces[i][0] + 1, faces_textures[i][0],
                                                          faces[i][1] + 1, faces_textures[i][1],
                                                          faces[i][2] + 1, faces_textures[i][2]) )
            return None
        def save_star_fbx(verts, joints, save_path, sample_STAR_path = '/hdd_ext/hdd2000/Working/dohae/Metown/utils/asset/STAR/basicModel_m_lbs_10_207_0_v1.0.2.fbx', fbx_path = '/home/dohae/FBX/build/Distrib/site-packages/fbx'):
            '''
            *** INPUT ***
                verts: [6890, 3]
                joints: [24, 3]
                save_path: path to save fbx file
                sample_STAR_path: sample STAR fbx file path
                fbx_path: path of fbx library
            *** OUTPUT ***
                None
            *** REQUIRED ***
                weights adjustment is not implemented yet.
            '''
            from .src.fbx import print_limb_child, store_bind_pose
            sys.path.append(fbx_path)
            import fbx
            import FbxCommon
            skeleton_name = ['m_avg_Pelvis', 'm_avg_L_Hip', 'm_avg_L_Knee', 'm_avg_L_Ankle', 'm_avg_L_Foot',
            'm_avg_R_Hip', 'm_avg_R_Knee', 'm_avg_R_Ankle', 'm_avg_R_Foot', 'm_avg_Spine1', 'm_avg_Spine2',
            'm_avg_Spine3', 'm_avg_Neck', 'm_avg_Head', 'm_avg_L_Collar', 'm_avg_L_Shoulder', 'm_avg_L_Elbow',
            'm_avg_L_Wrist', 'm_avg_L_Hand', 'm_avg_R_Collar', 'm_avg_R_Shoulder', 'm_avg_R_Elbow', 'm_avg_R_Wrist',
            'm_avg_R_Hand']
            parent = [0,1,2,3,0,5,6,7,0,9,10,11,12,11,14,15,16,17,11,19,20,21,22]
            J = np.asarray(joints)
            J_ = J.copy()
            J_[1:, :] = J[1:, :] - J[parent, :]
            skeleton = {}
            for i in range(len(skeleton_name)):
                skeleton[skeleton_name[i]] = J_[i]
            # Load sample fbx file and get mesh, skeleton
            manager = fbx.FbxManager.Create()
            importer = fbx.FbxImporter.Create(manager, 'myImporter')
            status = importer.Initialize(sample_STAR_path)
            if status == False:
                log('FbxImporter initialization failed.')
                log('Error: %s' % importer.GetLastErrorString())
                return
            scene = fbx.FbxScene.Create(manager, 'myScene')
            importer.Import(scene)

            rootNode = scene.GetRootNode()
            m_avg = rootNode.GetChild(0)
            m_avg_mesh = m_avg.GetNodeAttribute()
            m_avg_skeleton = rootNode.GetChild(1)
            # Modify the skeleton and vertices
            print_limb_child(fbx, m_avg_skeleton, skeleton) # skeleton
            for i in range(len(verts)): # vertices
                vert = verts[i]
                m_avg_mesh.SetControlPointAt(fbx.FbxVector4(vert[0],vert[1],vert[2],0), fbx.FbxVector4(0,0,0,0), i)
            # Set bind pose
            d1 = m_avg_mesh.GetDeformer(0)
            for i in range(d1.GetClusterCount()):
                c = d1.GetCluster(i)
                l = c.GetLink()
                lXMatrix = scene.GetAnimationEvaluator().GetNodeGlobalTransform(l)
                c.SetTransformLinkMatrix(lXMatrix)
            # Set skinning weights
            #for i1 in range(1, d1.GetClusterCount()):
            #    c = d1.GetCluster(i1)
            #    for n, i0 in enumerate(c.GetControlPointIndices()):
            #        STAR_weights[i0, i1 - 1] = c.GetControlPointWeights()[n]
            # Generate normal
            m_avg_mesh.GenerateNormals(True, True)
            # Remove blend shapes
            d2 = m_avg_mesh.GetDeformer(1) # blend shape
            blend_shapes = []#[verts_thin, verts_tall, verts_big_face, verts_trapezius]
            shapes_name = []#['thin', 'tall', 'big face', 'trapezius']
            while d2.GetBlendShapeChannelCount() > len(blend_shapes):
                d2.GetSrcObject(len(blend_shapes)).Destroy()
            # Save fbx
            exporter = fbx.FbxExporter.Create(manager, '')
            exporter.Initialize(save_path, -1, manager.GetIOSettings())
            exporter.Export(scene)
            scene.Destroy()
            manager.Destroy()
    class SMPLX:
        def __init__(self, gender = 'male', num_betas = 10, num_expression_coeffs = 10, batch_size = 1, device = 'cuda'):
            self.smplx = smplx.create(os.path.join(ROOT_DIR, 'smplx', 'models'), model_type='smplx',
                                 gender=gender,
                                 num_betas=num_betas,
                                 num_expression_coeffs=num_expression_coeffs,
                                 ext='npz',
                                 batch_size = batch_size).to(device)
            self.num_betas = num_betas
            self.num_expression_coeffs = num_expression_coeffs
            self.v_template = self.smplx.v_template.clone()
            log('SMPL-X initialized.')
        def forward(self, pose, shape, trans, disps = None, expression = None, device = 'cuda', to_numpy = False):
            '''
            INPUT
                pose: [batch, 63]
                shape: [batch, num_betas]
                disps: [N_V, 3]
            '''
            if shape.shape[1] < self.num_betas:
                shape = torch.cat((shape, torch.FloatTensor(np.zeros([len(pose), self.num_betas - shape.shape[1]])).to(device)), dim=1)
            if expression is None:
                expression = torch.randn([len(pose), self.num_expression_coeffs], dtype=torch.float32).to(device) * 0
            if disps is not None:
                self.smplx.v_template += disps
            output = self.smplx(betas=shape, expression=expression, body_pose=pose, transl = trans, return_verts=True)
            vertices = output.vertices
            joints = output.joints
            if to_numpy:
                vertices = vertices.detach().cpu().numpy()
                joints = joints.detach().cpu().numpy()
            self.joint_locs = joints[:, list(np.arange(22)) + [28, 43]]
            self.smplx.v_template = self.v_template.clone()
            return vertices
        def SMPLpose2MTHpose(self, pose):
            return pose
        def SMPLpose2SMPLXpose(self, smplpose):
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
        def SMPLXpose2SMPLpose(self, smplxpose):
            '''
            INPUT
                smplpose: [N, 21 * 3]
            OUTPUT
                smplxpose: [N, 24 * 3]
            '''
            smplxpose = smplxpose.reshape([-1, 21, 3])
            smplpose = smplxpose[:, [0, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 20,20]]
            smplpose[:, [0, -2, -1]] *= 0
            smplpose = smplpose.reshape([-1, 24 * 3])
            return smplpose
        def load_smplx(gender = 'male', num_betas = 10):
            smplx = smplx.create(model_folder=os.path.join(ROOT_DIR, 'smplx', 'models'), model_type='smplx',
                                 gender=gender,
                                 num_betas=num_betas,
                                 num_expression_coeffs=10,
                                 ext='npz')
            log('SMPL-X loaded completely.')
            return smplx
        # REQUIRED: NOT YET IMPLEMENTED
        def generate_random_smplx_param(N, seed = None, pose_rho = 0, pose_sig = 1, shape_rho = 0, shape_sig = 1, trans_rho = 0, trans_sig = 0, num_betas = 10):
            '''
            *** INPUT ***
                N: how many parames would be generated.
                seed: random seed (default: None)
                pose_rho: mean of pose
                pose_sig: standard deviation of pose (same way for shape and trans. trans sig is set to 0 in default, because trans is usually zero vector.)
            *** OUTPUT ***
                pose: [N, 72]
                shape: [N, 10]
                trans: [N, 3]
            '''
            if seed is not None:
                np.random.seed(seed)
            pose = np.random.normal(pose_rho, pose_sig, size=(N, 72))
            shape = np.random.normal(shape_rho, shape_sig, size=(N, num_betas))
            trans = np.random.normal(trans_rho, trans_sig, size=(N, 3))
            return pose, shape, trans
        def smplx_forward(smplx, pose, shape, trans, expression = None, disps = None, to_numpy = True, device = 'cuda'):
            '''
            *** INPUT ***
                smplx: smplx model
                pose: [N, 63]
                shape: [N, num_betas]
                trans: [N, 3]
                disps: [N, 10475, 3]
            *** OUTPUT ***
                vertices [N, 10475, 3]
            '''
            if type(pose) == np.ndarray:
                pose = torch.FloatTensor(pose)
            if type(shape) == np.ndarray:
                shape = torch.FloatTensor(shape)
            if type(trans) == np.ndarray:
                trans = torch.FloatTensor(trans)
            if disps is not None:
                if type(disps) == np.ndarray:
                    disps = torch.FloatTensor(disps)
                disps = disps.to('cuda')
                smplx.smplx.v_template += disps
            if expression is None:
                expression = torch.randn([len(pose), smplx.num_expression_coeffs], dtype=torch.float32) * 0

            output = smplx.smplx(betas=shape.to(device), expression=expression.to(device), body_pose=pose.to(device), transl = trans.to(device), return_verts=True)
            vertices = output.vertices
            if to_numpy:
                vertices = vertices.detach().cpu().numpy()
            smplx.smplx.v_template = smplx.v_template.clone()
            return vertices
        # REQUIRED: MORE FANCY WAY IS NEEDED
        def save_smplx_obj(verts, save_path):
            '''
            *** INPUT ***
                verts: [10475, 3]
                save_path: path to save obj file
            *** OUTPUT ***
                None
            '''
            obj_dict = Utils.OBJ.load_obj(os.path.join(ROOT_DIR, 'smplx', 'samples', 'smplx.obj'))
            obj_dict['verts'] = verts
            obj_dict['normals'] = []
            obj_dict['fNormals'] = []
            Utils.OBJ.save_obj(obj_dict, save_path, fToken = '/')
            return None
        # REQUIRED: NOT YET IMPLEMENTED
        def save_smplx_fbx(verts, joints, save_path, sample_STAR_path = '/hdd_ext/hdd2000/Working/dohae/Metown/utils/asset/STAR/basicModel_m_lbs_10_207_0_v1.0.2.fbx', fbx_path = '/home/dohae/FBX/build/Distrib/site-packages/fbx'):
            '''
            *** INPUT ***
                verts: [6890, 3]
                joints: [24, 3]
                save_path: path to save fbx file
                sample_STAR_path: sample STAR fbx file path
                fbx_path: path of fbx library
            *** OUTPUT ***
                None
            *** REQUIRED ***
                weights adjustment is not implemented yet.
            '''
            from .src.fbx import print_limb_child, store_bind_pose
            sys.path.append(fbx_path)
            import fbx
            import FbxCommon
            skeleton_name = ['m_avg_Pelvis', 'm_avg_L_Hip', 'm_avg_L_Knee', 'm_avg_L_Ankle', 'm_avg_L_Foot',
            'm_avg_R_Hip', 'm_avg_R_Knee', 'm_avg_R_Ankle', 'm_avg_R_Foot', 'm_avg_Spine1', 'm_avg_Spine2',
            'm_avg_Spine3', 'm_avg_Neck', 'm_avg_Head', 'm_avg_L_Collar', 'm_avg_L_Shoulder', 'm_avg_L_Elbow',
            'm_avg_L_Wrist', 'm_avg_L_Hand', 'm_avg_R_Collar', 'm_avg_R_Shoulder', 'm_avg_R_Elbow', 'm_avg_R_Wrist',
            'm_avg_R_Hand']
            parent = [0,1,2,3,0,5,6,7,0,9,10,11,12,11,14,15,16,17,11,19,20,21,22]
            J = np.asarray(joints)
            J_ = J.copy()
            J_[1:, :] = J[1:, :] - J[parent, :]
            skeleton = {}
            for i in range(len(skeleton_name)):
                skeleton[skeleton_name[i]] = J_[i]
            # Load sample fbx file and get mesh, skeleton
            manager = fbx.FbxManager.Create()
            importer = fbx.FbxImporter.Create(manager, 'myImporter')
            status = importer.Initialize(sample_STAR_path)
            if status == False:
                log('FbxImporter initialization failed.')
                log('Error: %s' % importer.GetLastErrorString())
                return
            scene = fbx.FbxScene.Create(manager, 'myScene')
            importer.Import(scene)

            rootNode = scene.GetRootNode()
            m_avg = rootNode.GetChild(0)
            m_avg_mesh = m_avg.GetNodeAttribute()
            m_avg_skeleton = rootNode.GetChild(1)
            # Modify the skeleton and vertices
            print_limb_child(fbx, m_avg_skeleton, skeleton) # skeleton
            for i in range(len(verts)): # vertices
                vert = verts[i]
                m_avg_mesh.SetControlPointAt(fbx.FbxVector4(vert[0],vert[1],vert[2],0), fbx.FbxVector4(0,0,0,0), i)
            # Set bind pose
            d1 = m_avg_mesh.GetDeformer(0)
            for i in range(d1.GetClusterCount()):
                c = d1.GetCluster(i)
                l = c.GetLink()
                lXMatrix = scene.GetAnimationEvaluator().GetNodeGlobalTransform(l)
                c.SetTransformLinkMatrix(lXMatrix)
            # Set skinning weights
            #for i1 in range(1, d1.GetClusterCount()):
            #    c = d1.GetCluster(i1)
            #    for n, i0 in enumerate(c.GetControlPointIndices()):
            #        STAR_weights[i0, i1 - 1] = c.GetControlPointWeights()[n]
            # Generate normal
            m_avg_mesh.GenerateNormals(True, True)
            # Remove blend shapes
            d2 = m_avg_mesh.GetDeformer(1) # blend shape
            blend_shapes = []#[verts_thin, verts_tall, verts_big_face, verts_trapezius]
            shapes_name = []#['thin', 'tall', 'big face', 'trapezius']
            while d2.GetBlendShapeChannelCount() > len(blend_shapes):
                d2.GetSrcObject(len(blend_shapes)).Destroy()
            # Save fbx
            exporter = fbx.FbxExporter.Create(manager, '')
            exporter.Initialize(save_path, -1, manager.GetIOSettings())
            exporter.Export(scene)
            scene.Destroy()
            manager.Destroy()
    class OBJ:
        def load_obj(file_path, silent = False, verts_only = False, to_numpy = False):
            '''
            *** INPUT ***
                file_path: obj file path
                silent = log or not
            *** OUTPUT ***
                obj_dict: dict
                    verts: [N_V, 3] (list)
                    colors: [N_V, 3] (list)
                    texture_coords: [N_T, 2] (or empty)
                    normals: [N_N, 3] (or empty)
                    fVerts: [N_F, 3]
                    fTexture_coords: [N_F, 3] (or empty)
                    fNormals: [N_F, 3] (or empty)
            *** CAUTION ***
                It can deal with triangle faces mesh, but cannot process meshes with more than 3 faces.
            '''
            verts = []
            colors = []
            texture_coords = []
            normals = []

            fVerts = []
            fTexture_coords = []
            fNormals = []

            has_normals = False
            has_texture_coords = False

            def set_has_normals(has_normals):
                if not has_normals:
                    has_normals = True
                    log('OBJ file has normals.', silent = silent)
                return True
            def set_has_textures(has_texture_coords):
                if not has_texture_coords:
                    has_texture_coords = True
                    log('OBJ file has texture coordinates.', silent = silent)
                return True

            face_split_token = None
            with open(file_path, 'r') as fr:
                log("OBJ file: [%s] now loading..."%file_path, silent = silent)
                while True:
                    line = fr.readline()
                    if not line: break
                    words = line.split(' ')
                    while '' in words:
                        words.remove('')
                    if words[0] == 'v':
                        verts.append([float(words[1]), float(words[2]), float(words[3])])
                        if len(words) >= 7: # with color
                            colors.append([float(words[4]), float(words[5]), float(words[6])])
                    elif words[0] == 'vt':
                        if verts_only: break
                        has_texture_coords = set_has_textures(has_texture_coords)
                        texture_coords.append([float(words[1]), float(words[2])])
                    elif words[0] == 'vn':
                        if verts_only: break
                        has_normals = set_has_normals(has_normals)
                        normals.append([float(words[1]), float(words[2]), float(words[3])])
                    elif words[0] == 'f':
                        if verts_only: break
                        if not (has_normals or has_texture_coords): face_split_token = 'ANYTHING'
                        if face_split_token is None: # get face split token
                            token_start = False
                            for character in words[1]:
                                if not token_start and (character >= '0' and character <= '9'):
                                    continue
                                elif not token_start:
                                    token_start = True
                                    face_split_token = character
                                elif not (character >= '0' and character <= '9'):
                                    face_split_token += character
                                else:
                                    break
                        #if len(words) != 4: error('load_obj only can process mesh with 3 faces. But, input mesh has %d faces'%len(words))
                        len_face = len(words) # number of vertices of the face + 1
                        fVerts.append([int(words[i].split(face_split_token)[0]) - 1 for i in range(1, len_face)])
                        if has_texture_coords:
                            fTexture_coords.append([int(words[i].split(face_split_token)[1]) - 1 for i in range(1, len_face)])
                        if has_texture_coords and has_normals:
                            fNormals.append([int(words[i].split(face_split_token)[2]) - 1 for i in range(1, len_face)])
                        if has_normals:
                            fNormals.append([int(words[i].split(face_split_token)[1]) - 1 for i in range(1, len_face)])
            obj_dict = {}
            if to_numpy:
                verts = np.asarray(verts)
                colors = np.asarray(colors)
                texture_coords = np.asarray(texture_coords)
                normals = np.asarray(normals)
                fVerts = np.asarray(fVerts)
                fTexture_coords = np.asarray(fTexture_coords)
                fNormalsfNormals = np.asarray(fNormals)
            obj_dict['verts'] = verts
            obj_dict['colors'] = colors
            obj_dict['texture_coords'] = texture_coords
            obj_dict['normals'] = normals
            obj_dict['fVerts'] = fVerts
            obj_dict['fTexture_coords'] = fTexture_coords
            obj_dict['fNormals'] = fNormals
            return obj_dict
        def save_obj(obj_dict, file_path, fToken = '/'):
            '''
            *** INPUT ***
                obj_dict: dict
                    verts: [N_V, 3] (list)
                    colors: [N_V, 3] (list)
                    texture_coords: [N_T, 2] (or empty)
                    normals: [N_N, 3] (or empty)
                    fVerts: [N_F, 3]
                    fTexture_coords: [N_F, 3] (or empty)
                    fNormals: [N_F, 3] (or empty)
                file_path: obj file path
            *** OUTPUT ***
                Nothing
            *** CAUTION ***
                It can deal with triangle faces mesh, but cannot process meshes with more than 3 faces.
            '''
            from datetime import datetime
            if not file_path.endswith('.obj'):
                file_path += '.obj'
            verts = obj_dict['verts']
            colors = obj_dict.get('colors', [])
            texture_coords = obj_dict.get('texture_coords', [])
            normals = obj_dict.get('normals', [])
            fVerts = obj_dict['fVerts']
            fTexture_coords = obj_dict.get('fTexture_coords', [])
            fNormals = obj_dict.get('fNormals', [])

            has_texture_coords = (len(texture_coords) != 0)
            has_normals = (len(normals) != 0)

            with open(file_path, 'w') as fp:
                fp.write('# MeTown mesh file generated at %s\n'%(datetime.now().strftime("%Y.%m.%d %H:%M:%S")))
                fp.write('# Copytight(c) %s All rights reserved\n'%datetime.now().strftime("%Y"))
                if len(colors) == 0:
                    for v in verts:
                        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
                else: # with colors
                    for i, v in enumerate(verts):
                        c = colors[i]
                        fp.write( 'v %f %f %f %f %f %f\n' % ( v[0], v[1], v[2], c[0], c[1], c[2]) )
                for t in texture_coords:
                    fp.write( 'vt %f %f\n' % ( t[0], t[1]) )
                for n in normals:
                    fp.write( 'vn %f %f %f\n' % ( n[0], n[1], n[2]) )
                if has_texture_coords and has_normals:
                    for i in range(len(fVerts)): # Faces are 1-based, not 0-based in obj files
                        len_face = len(fVerts[i]) # number of vertices in the face
                        fp.write('f')
                        for t in range(len_face):
                            fp.write( ' %d%s%d%s%d' % (fVerts[i][t] + 1, fToken, fTexture_coords[i][t] + 1, fToken, fNormals[i][t] + 1) )
                        fp.write('\n')
                elif has_texture_coords:
                    for i in range(len(fVerts)): # Faces are 1-based, not 0-based in obj files
                        len_face = len(fVerts[i]) # number of vertices in the face
                        fp.write('f')
                        for t in range(len_face):
                            fp.write( ' %d%s%d' % (fVerts[i][t] + 1, fToken, fTexture_coords[i][t] + 1) )
                        fp.write('\n')
                elif has_normals:
                    for i in range(len(fVerts)): # Faces are 1-based, not 0-based in obj files
                        len_face = len(fVerts[i]) # number of vertices in the face
                        fp.write('f')
                        for t in range(len_face):
                            fp.write( ' %d%s%d' % (fVerts[i][t] + 1, fToken, fNormals[i][t] + 1) )
                        fp.write('\n')
                else:
                    for i in range(len(fVerts)): # Faces are 1-based, not 0-based in obj files
                        len_face = len(fVerts[i]) # number of vertices in the face
                        fp.write('f')
                        for t in range(len_face):
                            fp.write( ' %d' % (fVerts[i][t] + 1) )
                        fp.write('\n')
            log('OBJ file [%s] save completed.'%file_path)
            return None

    class AgoraPose:
        # dataloader of agora pose data
        def __init__(self, root_path):
            '''
            *** INPUT ***
                 root_path: path of agora dataset root directory.
            '''
            self.root_path = root_path
            self.data_list = []
            for dir1 in os.listdir(root_path):
                for file_name in os.listdir(os.path.join(self.root_path, dir1)):
                    if file_name.endswith('.pkl'):
                        self.data_list.append(os.path.join(self.root_path, dir1, file_name))
            log('Agora pose dataset loaded completely: %s, length of data: %d'%(self.root_path, len(self.data_list)))
        def get_pose(self, i, to_numpy = True):
            '''
            *** INPUT ***
                i: index of data
            *** OUTPUT ***
                data {}
                    pose: [1, 72] (np)
                    path: data path
                    index: data index
            '''
            assert to_numpy, 'Agora pose data is originally torch.tensor type. Get tensor type of pose from Agroa should be implemented.'
            pose_data = pickle.load(open(self.data_list[i], 'rb'))
            if to_numpy:
                body_pose = pose_data['body_pose'].cpu().detach().numpy().reshape([1,69])
                root_pose = pose_data['root_pose'].cpu().detach().numpy().reshape([1,3])
                pose = np.concatenate([root_pose, body_pose], axis = 1)
            data = {}
            data['pose'] = pose
            data['path'] = self.data_list[i]
            data['index'] = i
            return data

class MTH:

    # MeTown Human (MTH) model.
    def __init__(self):
        log('MTH model loaded.')
    def multi_pose_data_load(self, data_root, silent = False, s_max = 10, p_max = 40):
        '''
        *** INPUT ***
            p_max: number of pose to load.
        '''
        self.data_root = data_root
        subject_list_ = os.listdir(self.data_root)
        subject_list = [sub for sub in subject_list_ if sub.startswith('subject')]
        subject_list.sort()
        self.data_list = [] # [I, J], J may be different for each subject i
        self.data = [] # [I, J, N_V, 3]
        for s_idx, subject in enumerate(subject_list):
            if s_max is not None and s_idx >= s_max: break
            pose_list_ = os.listdir(os.path.join(self.data_root, subject))
            pose_list = [pose for pose in pose_list_ if pose.startswith('pose')]
            pose_list.sort()
            self.data_list.append([])
            #self.data.append([])
            for p_idx, pose in enumerate(pose_list):
                if p_max is not None and p_idx >= p_max: break
                self.data_list[-1].append(os.path.join(self.data_root, subject, pose))
                #self.data.append(Utils.OBJ.load_obj(self.data_list[-1][-1], silent = True)['verts'])
        log('MTH data load completed.')
    def multi_shape_data_load(self, data_root, silent = False, s_max = 1000):
        '''
        *** INPUT ***
            s_max: number of shapes to load. If set to None, load max number of subjects.
        '''
        self.data_root = data_root
        subject_list_ = os.listdir(self.data_root)
        subject_list = [sub for sub in subject_list_ if sub.startswith('subject')]
        subject_list.sort()
        self.data_list = [] # [I, 1]
        self.data = [] # [I, 1, N_V, 3]
        for s_idx, subject in enumerate(subject_list):
            if s_max is not None and s_idx >= s_max: break
            pose_list_ = os.listdir(os.path.join(self.data_root, subject))
            pose_list = [pose for pose in pose_list_ if pose.endswith('.obj')]
            pose_list.sort()
            self.data_list.append([])
            for p_idx, pose in enumerate(pose_list):
                self.data_list[-1].append(os.path.join(self.data_root, subject, pose))
        log('MTH data load completed.')
    def data_read(self, s_idx, normalize = True):
        data = []
        for p_idx in range(len(self.data_list[s_idx])):
            data.append(np.asarray(Utils.OBJ.load_obj(self.data_list[s_idx][p_idx], silent = True, verts_only = True)['verts']))
            if normalize:
                data[-1] = data[-1] - data[-1].mean(axis=0)
            else:
                data[-1][:, 1] -= data[-1][:, 1].min()
        return torch.FloatTensor(data)
    def get_data_spec(self, template_path = '%s/MTH/Data/Template/SMPL/SMPLinfo.json'%ROOT_DIR, device = 'cpu', normalize = True):
        '''
        *** INPUT ***
            template_path: path of template
                template is dictionary json file
                    verts: [N_V, 3] (coordinates of vertices)
                    joints: [N_K, 3] (coordinates of joints)
                    joint_names: [N_k] (names of joints)
                    surround_idxs: [N_K, ?] (indices of vertex surrounding each joint. Their mean coordinate is initial position of the joint.)
                    weights: [N_V, N_K] (bleding weights of vertices)
                    parents: [N_K - 1] (parents of each joint (1st joint~))
                    Tmirror_idxs: [N_V] (indices of mirror to yz plane for each vertex)
                    Tmirror_idxs: [N_K] (indices of mirror to yz plane for each joint)
                    edges_list: [N_E, 2] (indices of vertices for each edge, N_E: #edge)
        *** PS ***
            See STAR_skeleton_exportor.blend.
            0-th joint should be root joint.
        '''
        with open(template_path, 'r') as reader:
            template_info = json.load(reader)
        self.Tverts = torch.FloatTensor(template_info['verts']).to(device)
        if normalize:
            verts_maen = self.Tverts.mean(axis=0)
            self.Tverts = self.Tverts - verts_maen
        self.Tjoints = torch.FloatTensor(template_info['joints']).to(device)
        if normalize:
            self.Tjoints = self.Tjoints - verts_maen
        self.joint_names = template_info['joint_names']
        self.surround_idxs = template_info['surround_idxs']
        self.Tweights = torch.FloatTensor(template_info['weights']).to(device)
        self.parents = torch.tensor(template_info['parents']).to(device)
        self.Tmirror_idxs = torch.tensor(template_info['mirror_vertices_idx_list']).to(device)
        self.Jmirror_idxs = torch.tensor(template_info['mirror_joints_idx_list']).to(device)
        self.edges_list = torch.tensor(template_info['edges_list']).to(device)
        self.N_V = len(self.Tverts) # number of vertices
        self.N_K = len(self.Tjoints) # number of joints
        self.N_KR = (self.N_K - 1) * 9 # Rodrigues fomula of joints
        self.N_S = len(self.data_list) # number of subjects
        log('data spec in [%s] load completed.'%template_path)
    def initialize_WPJ(self, device = 'cpu'):
        self.W = torch.FloatTensor(np.zeros([self.N_K, self.N_V])).to(device) # blend weights
        # self.W should become [4*3, N_V], because maximum number of joint per each vertex is 4.
        self.P = torch.FloatTensor(np.zeros([self.N_KR, self.N_V * 3])).to(device) # pose blend shapes
        #self.J = torch.FloatTensor(np.zeros([ self.N_V * 3, self.N_K * 3])).to(device) # joint regressor
    def quat2mat(self, quat):
        '''
        This function is get from STAR source.
        '''
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
        B = quat.size(0)
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
        return rotMat
    def pose2rodrigues(self, theta):
        '''
        This function is get from STAR source.
        *** INPUT ***
            theta: [N, 3] (torch)
        *** OUTPUT ***
            [N, 3, 3] (torch, rotation matrix)
        '''
        l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
        return self.quat2mat(quat)
    def get_optimizer(self, params, lr=1e-2):
        import torch.optim as optim
        '''
        *** INPUT ***
            params: list of parameters
        '''
        for param in params:
            param.requires_grad = True
        return optim.Adam( params, lr=lr)
    def get_initial_joints(self, Ts):
        '''
        *** INPUT ***
            Ts: [batch, N_V, 3], torch.tensor
        *** OUTPUT ***
            joints: [batch, N_K, 3], torch.tensor
        '''
        joints = []
        for k in range(self.N_K):
            joints.append(Ts[:, self.surround_idxs[k]].mean(axis=1).view(len(Ts), 1, 3)) # [batch, 1, 3]
        joints = torch.cat(joints, dim = 1) # [batch, N_K, 3]
        return joints
    def get_edge_loss(self, gt, pred):
        gt_start = gt[:, self.edges_list[:, 0]]
        gt_end = gt[:, self.edges_list[:, 1]]
        pred_start = pred[:, self.edges_list[:, 0]]
        pred_end = pred[:, self.edges_list[:, 1]]
        gt_edges = gt_end - gt_start # [N_V, 3]
        pred_edges = pred_end - pred_start # [N_V, 3]
        edge_loss = ((gt_edges - pred_edges) ** 2).sum() / len(gt)
        return edge_loss
    def get_loss_P(self, gt, pred, pred_J, pred_T):
        '''
        *** INPUT ***
            gt: [N_P, N_V, 3], torch.tensor
            pred: [N_P, N_V, 3], torch.tensor
            pred_J: [N_P, N_K, 3], torch.tensor (joints)
            pred_T: [N_P, N_V, 3], torch.tensor
        '''
        E_D = ((gt - pred) ** 2).sum() / len(gt)

        E_Y = ((pred_T[:, :, :1] + pred_T[:, self.Tmirror_idxs, :1]) ** 2).sum() / len(gt) # symmetry loss
        E_Y += ((pred_T[:, :, 1:] - pred_T[:, self.Tmirror_idxs, 1:]) ** 2).sum() / len(gt) # symmetry loss
        E_Y += ((pred_J[:, :, :1] + pred_J[:, self.Jmirror_idxs, :1]) ** 2).sum() / len(gt) # symmetry loss
        E_Y += ((pred_J[:, :, 1:] - pred_J[:, self.Jmirror_idxs, 1:]) ** 2).sum() / len(gt) # symmetry loss

        E_J = ((pred_J - self.get_initial_joints(pred_T)) ** 2).sum() / len(gt)

        # SMPL-specified loss. For a new template, it needs to be modified.
        pred_init_ratio = (pred_J[:, 13, 1] - pred_J[:, 0, 1]) / (self.Tjoints[13, 1] - self.Tjoints[0, 1]) # ratio of pred_J to Tjoints of length from head to pelvis
        E_JZ = (pred_J[:, [0,1,2,3,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 2] ** 2).sum() / len(gt) # Z value of joints should be close to 0
        E_JZ += ( (pred_J[:, 14, 1] - pred_J[:, 15, 1]) ** 2 + (pred_J[:, 14, 1] - pred_J[:, 16, 1]) ** 2 + (pred_J[:, 14, 1] - pred_J[:, 17, 1]) ** 2 + (pred_J[:, 18, 1] - pred_J[:, 19, 1]) ** 2 +
                    (pred_J[:, 14, 1] - pred_J[:, 20, 1]) ** 2 + (pred_J[:, 14, 1] - pred_J[:, 21, 1]) ** 2 + (pred_J[:, 14, 1] - pred_J[:, 22, 1]) ** 2 + (pred_J[:, 14, 1] - pred_J[:, 23, 1]) ** 2).sum() / len(gt) # the y value of arms should be same.
        E_JZ += ( (pred_J[:, 1, 0] - pred_J[:, 2, 0]) ** 2 + (pred_J[:, 1, 0] - pred_J[:, 3, 0]) ** 2 +  (pred_J[:, 1, 0] - pred_J[:, 4, 0]) ** 2 +
                    (pred_J[:, 5, 0] - pred_J[:, 6, 0]) ** 2 + (pred_J[:, 5, 0] - pred_J[:, 7, 0]) ** 2 + (pred_J[:, 5, 0] - pred_J[:, 8, 0]) ** 2).sum() / len(gt) # the x value of legs should be same.
        #E_JZ += (pred_J[:, 0, 0] ** 2).sum() / len(gt) # x of root joint to zero.
        diff_ankley_footy = (self.Tjoints[3, 1] - self.Tjoints[4, 1]) * pred_init_ratio
        E_JZ += ( (pred_J[:, 3, 1] - pred_J[:, 4, 1] - diff_ankley_footy) ** 2 + (pred_J[:, 7, 1] - pred_J[:, 8, 1] - diff_ankley_footy) ** 2 ).sum() / len(gt) # the y value of foot and ankle should be same.

        E_P = (self.P ** 2).sum()
        E_W = (self.W ** 2).sum()
        return E_D, E_Y, E_J, E_JZ, E_P, E_W
    def normalize_pose(self, posed_vs, epochs = 1000, device = 'cuda', obj_save_dir = './example_obj2_shape', obj_save_interval = None):
        '''
        *** INPUT ***
            posed_vs: [batch, 1, N_V * 3]
        *** OUTPUT ***
            T_vs: [batch, N_V * 3]
        '''
        self.T_S = torch.FloatTensor(np.zeros([self.N_S, self.N_V, 3])).to(device)
        self.theta = torch.FloatTensor(np.zeros([self.N_S, len(self.data_list[0]), self.N_K, 3])).to(device)
        optimizer1 = self.get_optimizer([self.theta], lr=1e-2)
        optimizer2 = self.get_optimizer([self.T_S], lr=1e-3)
        optimizer3 = self.get_optimizer([self.theta, self.T_S], lr=1e-3)

        self.phase = 0
        epochs_ = []
        phases_ = []
        for epoch, phase in epochs:
            epochs_.append(epoch)
            phases_.append(phase)
        epoch = 0
        optimizer = None

        for e_idx, epochs_partial in enumerate(epochs_):
            if phases_[e_idx] == 1:
                optimizer = optimizer1
                self.phase = 1
            elif phases_[e_idx] == 2:
                optimizer = optimizer2
                self.phase = 2
            elif phases_[e_idx] == 3:
                optimizer = optimizer3
                self.phase = 3

            for _ in range(epochs_partial):
                epoch_start = time.time()

                N_P = len(self.data_list[0]) # 1
                T = self.Tverts.tile(self.N_S, 1, 1) + self.T_S # [N_S, N_V, 3]
                T = T.view(self.N_S, 1, self.N_V * 3).tile(1, N_P, 1).view(self.N_S * N_P, self.N_V * 3) # [N_S * 1, N_V * 3]
                R = self.pose2rodrigues(self.theta.contiguous().view(-1, 3)).view(self.N_S * N_P, self.N_K, 3, 3) # [N_S * 1, N_K, 3, 3]
                T_pose = T + torch.matmul(self.P.T, R[:, 1:].view(-1, self.N_KR).T).view(self.N_V * 3, self.N_S * N_P).T # [N_S * 1, N_V * 3], add pose blend shapes
                T_pose = T_pose.view(self.N_S * N_P, self.N_V, 3) # [N_S * 1, N_V, 3]
                T = T.view(self.N_S * N_P, self.N_V, 3) # [N_S * 1, N_V, 3]

                #J = torch.matmul(self.J.T, T.view(self.N_S * N_P, self.N_V * 3).T).T.view(self.N_S, self.N_K, 3) # [N_S * 1, N_K, 3]
                J = self.J_regress(T.view(self.N_S, self.N_V * 3)).view(self.N_S, self.N_K, 3) # [N_S * 1, N_K, 3]
                J_ = J.clone() # [N_S * N_P, N_K, 3]
                J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parents, :]
                G_ = torch.cat([R, J_[:, :, :, None]], dim=-1) # [N_S * N_P, N_K, 3, 4]
                pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(self.N_S * N_P, self.N_K, -1, -1) # [N_S * N_P, N_K, 3, 4]
                G_ = torch.cat([G_, pad_row], dim=2) # [N_S * N_P, N_K, 4, 4]
                G = [G_[:, 0].clone()] # [1, N_S * N_P, 4, 4]
                for i in range(1, self.N_K):
                    G.append(torch.matmul(G[self.parents[i - 1]], G_[:, i, :, :])) # [N_K, N_S * N_P, 4, 4]
                G = torch.stack(G, dim=1) # [N_P, N_K, 4, 4]
                rest = torch.cat([J, torch.zeros(self.N_S * N_P, self.N_K, 1).to(device)], dim=2).view(self.N_S * N_P, self.N_K, 4, 1) # [N_S * N_P, N_K, 4] -> [N_S * N_P, N_K, 4, 1]

                zeros = torch.zeros(self.N_S * N_P, self.N_K, 4, 3).to(device)
                rest = torch.cat([zeros, rest], dim=-1) # [N_S * N_P, N_K, 4, 4]
                rest = torch.matmul(G, rest) # [N_S * N_P, N_K, 4, 4]
                G = G - rest
                T_ = torch.matmul(self.Tweights + self.W.T, G.permute(1, 0, 2, 3).contiguous().view(self.N_K, -1)).view(self.N_V, self.N_S * N_P, 4,4).transpose(0, 1) # [N_S * N_P, N_V, 4, 4]
                rest_shape_h = torch.cat([T_pose, torch.ones_like(T_pose)[:, :, [0]]], dim=-1) # [N_S * N_P, N_V, 4]
                v = torch.matmul(T_, rest_shape_h[:, :, :, None])[:, :, :3, 0] # [N_S * N_P, N_V, 3]

                gt_v = posed_vs.view(self.N_S * N_P, self.N_V, 3) # [N_S * 1, N_V, 3]
                E_D, E_Y, E_J, E_JZ, E_P, E_W = self.get_loss_P(gt_v, v, J, T)
                E_D = E_D
                E_Y = E_Y * 100
                E_J = E_J * 100
                E_JZ = E_JZ * 100
                E_P = E_P * 25
                E_W = E_W
                edge_loss = self.get_edge_loss(gt_v, v) * 100
                #loss = E_D + E_Y + E_P + E_W + E_JZ + E_J
                loss = E_D + E_Y + E_P + E_W + E_J + E_JZ + edge_loss
                #loss = E_D + edge_loss
                if self.phase == 1:
                    loss = edge_loss
                #if self.phase == 3:
                #    loss += (E_J + E_JZ)
                optimizer.zero_grad()
                (loss).backward()
                optimizer.step()

                if obj_save_dir is not None and ( epoch == sum(epochs_)-1 or (obj_save_interval != None and (epoch+1) % obj_save_interval == 1)):
                    os.makedirs(obj_save_dir, exist_ok = True)
                    os.makedirs(os.path.join(obj_save_dir, 'epoch_%04d'%(epoch+1)), exist_ok = True)
                    p_idx = 0
                    for s_idx in range(min(5, self.N_S)):
                        Utils.STAR.save_star_obj(gt_v[s_idx * N_P + p_idx].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_gt_E%04d_S%04d_P%04d.obj'%(epoch+1, epoch+1, s_idx+1, p_idx+1)))
                        Utils.STAR.save_star_obj(v[s_idx * N_P + p_idx].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_pred_E%04d_S%04d_P%04d.obj'%(epoch+1, epoch+1, s_idx+1, p_idx+1)))
                        Utils.STAR.save_star_obj(T[s_idx * N_P].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_pred_E%04d_S%04d_T.obj'%(epoch+1, epoch+1, s_idx+1)))
                        Utils.STAR.save_star_fbx(T[s_idx * N_P].cpu().detach().numpy(), J[s_idx * N_P].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_pred_E%04d_S%04d_T.fbx'%(epoch+1, epoch+1, s_idx+1)))
                    #Utils.STAR.save_star_obj(self.Tverts.cpu().detach().numpy(), './example_obj/sample_pred_%d_%d.obj'%(epoch, s_idx))
                log('%d/%s  step (phase %d) (%.3fs): total_loss - %.2f D - %.2f Y - %.2f J - %.2f JZ - %.2f P - %.2f W - %.2f E - %.2f'%(epoch+1, sum(epochs_), self.phase, time.time()-epoch_start, loss.item(), E_D.item(), E_Y.item(), E_J.item(), E_JZ.item(), E_P.item(), E_W.item(), edge_loss.item()))
                epoch+=1

        return T
    def train_S(self, epochs, N_B = 10, obj_save_interval = None, obj_save_dir = './example_obj2_shape', device = 'cuda'):
        '''
        *** INPUT ***
            N_B: dimension of beta (shape)

        1. load multi-shape data
        2. pose normalization
        3. PCA
        '''
        log('SHAPE MODEL: data loading...')
        self.data = []
        for s_idx in range(self.N_S):
            self.data.append(self.data_read(s_idx).view(1, -1, self.N_V, 3).to(device)) # [1, 1, N_V, 3]
            if (s_idx + 1) % 10 == 0:
                log('data loading... %d/%d'%(s_idx+1, self.N_S))
        self.data = torch.cat(self.data, dim = 0) # [N_S, 1, N_V, 3]
        self.T_data = self.normalize_pose(self.data, epochs = epochs, device = device, obj_save_dir = obj_save_dir, obj_save_interval = obj_save_interval) # [N_S, N_V, 3]
        self.T_data = self.T_data - self.T_data.mean(axis = 1).view(self.N_S, 1, 3)
        self.T_data = self.T_data.view(self.N_S, self.N_V * 3)
        self.T_data_np = self.T_data.detach().cpu().numpy()
        # normalize position
        self.T_data_np = self.T_data_np - self.T_data_np.mean(axis=1, keepdims=True)
        self.T_data_np = self.T_data_np.reshape([self.N_S, self.N_V * 3])

        '''
        Train S (self.S [N_V * 3, N_B]).
            [N_V, 3] -> [N_B]
        I implemented it with gradient descent, but it may be calculated through closed form solution.
        '''
        from sklearn.decomposition import PCA
        self.N_B = N_B
        self.pca = PCA(n_components=self.N_B)
        pc = self.pca.fit_transform(self.T_data_np) # principal components
        self.S = []
        for component in self.pca.components_:
            self.S.append(torch.FloatTensor(component).to(device).view(len(component), 1))
        self.S = torch.cat(self.S, dim = 1).to(device) # [N_V * 3, N_B]
        self.S_mean = torch.FloatTensor(self.pca.mean_).view(1, self.N_V * 3).to(device)

    def train_P(self, epochs, device = 'cuda', seed = 0, obj_save_dir = './example_obj', obj_save_interval = None):
        '''
            Three phase (failed):
                1. Train T_S, J_S, and theta for each subject in only one pose.
                2. Train theta for each subject in all poses.
                3. Train T_S, J_S, theta for each subject in all poses.
                4. Train all parameters
            phase (failed):
                1. Train theta for each subject in all poses. (learning rate: 1e-1)
                2. Train T_S, J_S, and theta for each subject in all poses. (learning rate: 1e-1)
                3. Train W and P. (learning rate: 1e-3)
                4. Train all parameters. (learning rate: 1e-3)
            phase:
                1. Train theta for each subject in all poses. (learning rate: 1e-1)
                2. Train T_S and J_S for each subject in all poses. (learning rate: 1e-1)
                    Iterate 1st and 2nd phase [iter1] times.
                3. Train W and P. (learning rate: 1e-3)
                4. Train all parameters. (learning rate: 1e-3)
        '''
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        log('POSE MODEL: data loading...')
        self.data = []
        for s_idx in range(self.N_S):
            self.data.append(self.data_read(s_idx).view(1, -1, self.N_V, 3).to(device)) # [1, N_P, N_V, 3]
            if (s_idx + 1) % 10 == 0:
                log('data loading... %d/%d'%(s_idx+1, self.N_S))
        self.data = torch.cat(self.data, dim = 0) # [N_S, N_P, N_V, 3]
        # pose corrective model
        self.get_data_spec(device = device)
        self.initialize_WPJ(device = device)

        self.T_S = torch.FloatTensor(np.zeros([self.N_S, self.N_V, 3])).to(device) # [N_S, N_V, 3], T pose vertices for each subject
        self.J_S = torch.FloatTensor(np.zeros([self.N_S, self.N_K, 3])).to(device) # [N_S, N_K, 3], T pose joints for each subject
        # REQUIRED: for now, we assumed that number of pose is all same among subjects. But, it is not always true in real data.
        self.theta = torch.FloatTensor(np.zeros([self.N_S, len(self.data_list[0]), self.N_K, 3])).to(device) # [N_S, N_pose, N_K, 3], pose parameter
        #self.theta[:, :, 3] += 0.1
        optimizer1 = self.get_optimizer([self.theta], lr=1e-2)
        optimizer2 = self.get_optimizer([self.T_S, self.J_S], lr=1e-3)
        optimizer3 = self.get_optimizer([self.J_S], lr=1e-3)
        optimizer4 = self.get_optimizer([self.P, self.W], lr=1e-4)

        self.phase = 0
        epochs_ = []
        phases_ = []
        for epoch, phase in epochs:
            epochs_.append(epoch)
            phases_.append(phase)
        epoch = 0
        optimizer = None
        for e_idx, epochs_partial in enumerate(epochs_):
            if phases_[e_idx] == 1:
                optimizer = optimizer1
                self.phase = 1
            elif phases_[e_idx] == 2:
                optimizer = optimizer2
                self.phase = 2
            elif phases_[e_idx] == 3:
                optimizer = optimizer3
                self.phase = 3
            elif phases_[e_idx] == 4:
                optimizer = optimizer4
                self.phase = 4

            for _ in range(epochs_partial):
                epoch_start = time.time()
                ''' all subject version '''
                N_P = len(self.data_list[0])
                T = self.Tverts.tile(self.N_S, 1, 1) + self.T_S # [N_S, N_V, 3]
                T = T.view(self.N_S, 1, self.N_V * 3).tile(1, N_P, 1).view(self.N_S * N_P, self.N_V * 3) # [N_S * N_P, N_V * 3]
                R = self.pose2rodrigues(self.theta.contiguous().view(-1, 3)).view(self.N_S * N_P, self.N_K, 3, 3) # [N_S * N_P, N_K, 3, 3]
                T_pose = T + torch.matmul(self.P.T, R[:, 1:].view(-1, self.N_KR).T).view(self.N_V * 3, self.N_S * N_P).T # [N_S * N_P, N_V * 3], add pose blend shapes
                T_pose = T_pose.view(self.N_S * N_P, self.N_V, 3) # [N_S * N_P, N_V, 3]
                T = T.view(self.N_S * N_P, self.N_V, 3) # [N_S * N_P, N_V, 3]

                J = self.J_S.view(self.N_S, 1, self.N_K, 3).tile(1, N_P, 1, 1) # [N_S, N_P, N_K, 3]
                J = J.view(self.N_S * N_P, self.N_K, 3) # [N_S * N_P, N_K, 3]
                J = J + self.Tjoints.tile(self.N_S * N_P, 1, 1)
                #J = J + self.get_initial_joints(T) # [N_S * N_P, N_K, 3]
                J_ = J.clone() # [N_S * N_P, N_K, 3]
                J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parents, :]
                G_ = torch.cat([R, J_[:, :, :, None]], dim=-1) # [N_S * N_P, N_K, 3, 4]
                pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(self.N_S * N_P, self.N_K, -1, -1) # [N_S * N_P, N_K, 3, 4]
                G_ = torch.cat([G_, pad_row], dim=2) # [N_S * N_P, N_K, 4, 4]
                G = [G_[:, 0].clone()] # [1, N_S * N_P, 4, 4]
                for i in range(1, self.N_K):
                    G.append(torch.matmul(G[self.parents[i - 1]], G_[:, i, :, :])) # [N_K, N_S * N_P, 4, 4]
                G = torch.stack(G, dim=1) # [N_P, N_K, 4, 4]
                rest = torch.cat([J, torch.zeros(self.N_S * N_P, self.N_K, 1).to(device)], dim=2).view(self.N_S * N_P, self.N_K, 4, 1) # [N_S * N_P, N_K, 4] -> [N_S * N_P, N_K, 4, 1]

                zeros = torch.zeros(self.N_S * N_P, self.N_K, 4, 3).to(device)
                rest = torch.cat([zeros, rest], dim=-1) # [N_S * N_P, N_K, 4, 4]
                rest = torch.matmul(G, rest) # [N_S * N_P, N_K, 4, 4]
                G = G - rest
                T_ = torch.matmul(self.Tweights + self.W.T, G.permute(1, 0, 2, 3).contiguous().view(self.N_K, -1)).view(self.N_V, self.N_S * N_P, 4,4).transpose(0, 1) # [N_S * N_P, N_V, 4, 4]
                rest_shape_h = torch.cat([T_pose, torch.ones_like(T_pose)[:, :, [0]]], dim=-1) # [N_S * N_P, N_V, 4]
                v = torch.matmul(T_, rest_shape_h[:, :, :, None])[:, :, :3, 0] # [N_S * N_P, N_V, 3]

                gt_v = self.data.view(self.N_S * N_P, self.N_V, 3) # [N_S, N_P, N_V, 3]
                E_D, E_Y, E_J, E_JZ, E_P, E_W = self.get_loss_P(gt_v, v, J, T)
                E_D = E_D
                E_Y = E_Y * 100
                E_J = E_J * 100
                E_JZ = E_JZ * 100
                E_P = E_P * 25
                E_W = E_W
                edge_loss = self.get_edge_loss(gt_v, v) * 100
                #loss = E_D + E_Y + E_P + E_W + E_JZ + E_J
                loss = E_D + E_Y + E_P + E_W + E_J + E_JZ + edge_loss
                #if self.phase == 1:
                #    loss = edge_loss
                #if self.phase == 3:
                #    loss += (E_J + E_JZ)
                optimizer.zero_grad()
                (loss).backward()
                optimizer.step()

                if obj_save_dir is not None and ( epoch == sum(epochs_)-1 or (obj_save_interval != None and (epoch+1) % obj_save_interval == 1)):
                    os.makedirs(obj_save_dir, exist_ok = True)
                    os.makedirs(os.path.join(obj_save_dir, 'epoch_%04d'%(epoch+1)), exist_ok = True)
                    for s_idx in range(min(5, self.N_S)):
                        for p_idx in range(min(3, N_P)):
                            Utils.STAR.save_star_obj(gt_v[s_idx * N_P + p_idx].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_gt_E%04d_S%04d_P%04d.obj'%(epoch+1, epoch+1, s_idx+1, p_idx+1)))
                            Utils.STAR.save_star_obj(v[s_idx * N_P + p_idx].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_pred_E%04d_S%04d_P%04d.obj'%(epoch+1, epoch+1, s_idx+1, p_idx+1)))
                        Utils.STAR.save_star_obj(T[s_idx * N_P].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_pred_E%04d_S%04d_T.obj'%(epoch+1, epoch+1, s_idx+1)))
                        Utils.STAR.save_star_fbx(T[s_idx * N_P].cpu().detach().numpy(), J[s_idx * N_P].cpu().detach().numpy(), os.path.join(obj_save_dir, 'epoch_%04d/sample_pred_E%04d_S%04d_T.fbx'%(epoch+1, epoch+1, s_idx+1)))
                    #Utils.STAR.save_star_obj(self.Tverts.cpu().detach().numpy(), './example_obj/sample_pred_%d_%d.obj'%(epoch, s_idx))
                log('%d/%s  step (phase %d) (%.3fs): total_loss - %.2f D - %.2f Y - %.2f J - %.2f JZ - %.2f P - %.2f W - %.2f E - %.2f'%(epoch+1, sum(epochs_), self.phase, time.time()-epoch_start, loss.item(), E_D.item(), E_Y.item(), E_J.item(), E_JZ.item(), E_P.item(), E_W.item(), edge_loss.item()))
                epoch += 1
    def train_J(self, epochs, device = 'cuda'):
        '''
        Train J_regressor (self.J [N_V * 3, N_K * 3]).
            [N_V, 3] -> [N_K, 3]
        I implemented it with gradient descent, but it may be calculated through closed form solution.
        22.03.07: I implemented it through least square.
        '''
        T = self.Tverts.tile(self.N_S, 1, 1) + self.T_S # [N_S, N_V, 3]
        T = T.view(self.N_S, self.N_V * 3)
        J = self.J_S.view(self.N_S, self.N_K, 3) + self.Tjoints.tile(self.N_S, 1, 1) # [N_S, N_K, 3]
        #J = self.J_S.view(self.N_S, self.N_K, 3) + self.get_initial_joints(T.view(self.N_S, self.N_V, 3)) # [N_S, N_K, 3]
        J = J.view(self.N_S, self.N_K * 3)

        '''
        optimizer = self.get_optimizer([self.J], lr=1e-4)
        for epoch in range(epochs):
            eps = torch.normal(mean = 0, std = 0.1, size=(self.N_S, 1, 3)).to(device)
            J_pred = torch.matmul(self.J.T, (T + eps.tile(1, self.N_V, 1).view(self.N_S, self.N_V * 3)).T).T
            loss = ( (J_pred - (J+eps.tile(1, self.N_K, 1).view(self.N_S, self.N_K * 3))) ** 2 ).sum() / len(J_pred)
            optimizer.zero_grad()
            (loss).backward()
            optimizer.step()
            if epoch == 0 or epoch == epochs-1:
                log('J_regressor %d step: total_loss - %.2f'%(epoch+1, loss.item()))
        '''
        from sklearn.linear_model import LinearRegression
        J_regressor = LinearRegression()
        T_np = T.detach().cpu().numpy() # [N_S, N_V * 3]
        J_np = J.detach().cpu().numpy() # [N_S, N_K * 3]
        J_regressor.fit(T_np, J_np)
        self.J_W = torch.FloatTensor(J_regressor.coef_).to(device)
        self.J_b = torch.FloatTensor(J_regressor.intercept_).to(device)
    def J_regress(self, T):
        '''
        *** INPUT ***
            T: [batch, N_V, 3]
        *** OUTPUT ***
            J: [batch, N_K, 3]
        '''
        return torch.matmul(T, self.J_W.T) + self.J_b.tile(1,1)

    def model_save_P(self, save_path):
        '''
        Save parameters
        {
            W: blending weights regressor
            J_W: joint regressor weight
            J_n: joint regressor bias
            P: pose corrective model
        }
        '''
        torch.save({
            'W': self.W,
            'J_W': self.J_W,
            'J_b': self.J_b,
            'P': self.P
        }, save_path)
    def model_load_P(self, load_path):
        '''
        Load parameters
        {
            W: blending weights regressor
            J_W: joint regressor weight
            J_n: joint regressor bias
            P: pose corrective model
        }
        '''
        checkpoint = torch.load(load_path)
        self.W = checkpoint['W']
        self.J_W = checkpoint['J_W']
        self.J_b = checkpoint['J_b']
        self.P = checkpoint['P']
    def model_save_S(self, save_path):
        '''
        Save parameters
        {
            W: blending weights regressor
            J_W: joint regressor
            J_b: joint regressor
            P: pose corrective model
            S: shape regressor
            S_mean: shape mean
            N_B: number of shapes
            N_K: number of joints
            N_KR: number of Rodrigues rotation dimension
            N_V: number of vertices
            parents: parents of joints
            Tweights: initial weights
        }
        '''
        torch.save({
            'W': self.W,
            'J_W': self.J_W,
            'J_b': self.J_b,
            'P': self.P,
            'S': self.S,
            'S_mean': self.S_mean,
            'N_B': self.N_B,
            'N_K': self.N_K,
            'N_KR': self.N_KR,
            'N_V': self.N_V,
            'parents': self.parents,
            'Tweights': self.Tweights,
        }, save_path)
    def model_load_S(self, load_path):
        '''
        Load parameters
        {
            W: blending weights regressor
            J_W: joint regressor
            J_b: joint regressor
            P: pose corrective model
            S: shape regressor
            S_mean: shape mean
            N_B: number of shapes
            N_K: number of joints
            N_KR: number of Rodrigues rotation dimension
            N_V: number of vertices
            parents: parents of joints
            Tweights: initial weights
        }
        '''
        checkpoint = torch.load(load_path)
        self.W = checkpoint['W']
        self.J_W = checkpoint['J_W']
        self.J_b = checkpoint['J_b']
        self.P = checkpoint['P']
        self.S = checkpoint['S']
        self.S_mean = checkpoint['S_mean']
        self.N_B = checkpoint['N_B']
        self.N_K = checkpoint['N_K']
        self.N_KR = checkpoint['N_KR']
        self.N_V = checkpoint['N_V']
        self.parents = checkpoint['parents']
        self.Tweights = checkpoint['Tweights']
    def SMPLpose2MTHpose(self, pose):
        '''
        *** INPUT ***
            pose: [batch, 24*3]
        *** DESCRIPTION ***
            # ['m_avg_Pelvis', 'm_avg_L_Hip', 'm_avg_L_Knee', 'm_avg_L_Ankle', 'm_avg_L_Foot', 'm_avg_R_Hip', 'm_avg_R_Knee', 'm_avg_R_Ankle', 'm_avg_R_Foot', 'm_avg_Spine1', 'm_avg_Spine2', 'm_avg_Spine3', 'm_avg_Neck', 'm_avg_Head', 'm_avg_L_Collar', 'm_avg_L_Shoulder', 'm_avg_L_Elbow', 'm_avg_L_Wrist', 'm_avg_L_Hand', 'm_avg_R_Collar', 'm_avg_R_Shoulder', 'm_avg_R_Elbow', 'm_avg_R_Wrist', 'm_avg_R_Hand']
            # to
            # ['m_avg_Pelvis', 'm_avg_L_Hip', 'm_avg_R_Hip', 'm_avg_Spine1', 'm_avg_L_Knee', 'm_avg_R_Knee', 'm_avg_Spine2', 'm_avg_L_Ankle', 'm_avg_R_Ankle', 'm_avg_Spine3', 'm_avg_L_Foot', 'm_avg_R_Foot', 'm_avg_Neck', 'm_avg_L_Collar', 'm_avg_R_Collar', 'm_avg_Head', 'm_avg_L_Shoulder', 'm_avg_R_Shoulder', 'm_avg_L_Elbow', 'm_avg_R_Elbow', 'm_avg_L_Wrist', 'm_avg_R_Wrist', 'm_avg_L_Hand', 'm_avg_R_Hand']
        '''
        pose = pose.view(len(pose), 24, 3)
        MTHpose = pose * 0
        MTHpose[:, 0] = pose[:, 0]
        MTHpose[:, 1] = pose[:, 1]
        MTHpose[:, 2] = pose[:, 4]
        MTHpose[:, 3] = pose[:, 7]
        MTHpose[:, 4] = pose[:, 10]
        MTHpose[:, 5] = pose[:, 2]
        MTHpose[:, 6] = pose[:, 5]
        MTHpose[:, 7] = pose[:, 8]
        MTHpose[:, 8] = pose[:, 11]
        MTHpose[:, 9] = pose[:, 3]
        MTHpose[:, 10] = pose[:, 6]
        MTHpose[:, 11] = pose[:, 9]
        MTHpose[:, 12] = pose[:, 12]
        MTHpose[:, 13] = pose[:, 15]
        MTHpose[:, 14] = pose[:, 13]
        MTHpose[:, 15] = pose[:, 16]
        MTHpose[:, 16] = pose[:, 18]
        MTHpose[:, 17] = pose[:, 20]
        MTHpose[:, 18] = pose[:, 22]
        MTHpose[:, 19] = pose[:, 14]
        MTHpose[:, 20] = pose[:, 17]
        MTHpose[:, 21] = pose[:, 19]
        MTHpose[:, 22] = pose[:, 21]
        MTHpose[:, 23] = pose[:, 23]
        return MTHpose.view(-1, 72)
    def forward(self, pose, shape, trans, device = 'cuda'):
        '''
        *** INPUT ***
            pose: [batch, N_K * 3] (torch.tensor)
            shape: [batch, N_B]
            trans: [batch, 3]
        *** OUTPUT ***
            vertices: [batch, N_V, 3]
        '''
        self.N_S = len(shape)
        N_P = 1

        T = torch.matmul(self.S, shape.T).T + self.S_mean # [N_S, N_V * 3]
        T = T.view(self.N_S, 1, self.N_V * 3).tile(1, N_P, 1).view(self.N_S * N_P, self.N_V * 3) # [N_S * 1, N_V * 3]
        R = self.pose2rodrigues(pose.contiguous().view(-1, 3)).view(self.N_S * N_P, self.N_K, 3, 3) # [N_S * 1, N_K, 3, 3]
        T_pose = T + torch.matmul(self.P.T, R[:, 1:].view(-1, self.N_KR).T).view(self.N_V * 3, self.N_S * N_P).T # [N_S * 1, N_V * 3], add pose blend shapes
        T_pose = T_pose.view(self.N_S * N_P, self.N_V, 3) # [N_S * 1, N_V, 3]
        T = T.view(self.N_S * N_P, self.N_V, 3) # [N_S * 1, N_V, 3]

        #J = torch.matmul(self.J.T, T.view(self.N_S * N_P, self.N_V * 3).T).T.view(self.N_S, self.N_K, 3) # [N_S * 1, N_K, 3]
        J = self.J_regress(T.view(self.N_S, self.N_V * 3)).view(self.N_S, self.N_K, 3) # [N_S * 1, N_K, 3]
        J_ = J.clone() # [N_S * N_P, N_K, 3]
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parents, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1) # [N_S * N_P, N_K, 3, 4]
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(self.N_S * N_P, self.N_K, -1, -1) # [N_S * N_P, N_K, 3, 4]
        G_ = torch.cat([G_, pad_row], dim=2) # [N_S * N_P, N_K, 4, 4]
        G = [G_[:, 0].clone()] # [1, N_S * N_P, 4, 4]
        for i in range(1, self.N_K):
            G.append(torch.matmul(G[self.parents[i - 1]], G_[:, i, :, :])) # [N_K, N_S * N_P, 4, 4]
        G = torch.stack(G, dim=1) # [N_P, N_K, 4, 4]
        rest = torch.cat([J, torch.zeros(self.N_S * N_P, self.N_K, 1).to(device)], dim=2).view(self.N_S * N_P, self.N_K, 4, 1) # [N_S * N_P, N_K, 4] -> [N_S * N_P, N_K, 4, 1]

        zeros = torch.zeros(self.N_S * N_P, self.N_K, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1) # [N_S * N_P, N_K, 4, 4]
        rest = torch.matmul(G, rest) # [N_S * N_P, N_K, 4, 4]
        G = G - rest
        T_ = torch.matmul(self.Tweights + self.W.T, G.permute(1, 0, 2, 3).contiguous().view(self.N_K, -1)).view(self.N_V, self.N_S * N_P, 4,4).transpose(0, 1) # [N_S * N_P, N_V, 4, 4]
        rest_shape_h = torch.cat([T_pose, torch.ones_like(T_pose)[:, :, [0]]], dim=-1) # [N_S * N_P, N_V, 4]
        v = torch.matmul(T_, rest_shape_h[:, :, :, None])[:, :, :3, 0] # [N_S * N_P, N_V, 3]
        v = v + trans.view(len(v), 1, 3)
        self.joint_locs = J + trans.view(len(v), 1, 3) # [N_S, N_K, 3]
        return v
