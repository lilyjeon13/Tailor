import numpy as np
import sys
STAR_DIR = '../STAR'
sys.path.append(STAR_DIR)
from pytorch.star import STAR as STAR_
import torch
import os
import json

def log(message, silent = False):
    if not silent:
        print('>>>', message)

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


class STAR:
    def __init__(self):
        self.star = STAR_(gender = 'male')
        log('Star initialized.')
    def load_star():
        star = STAR_(gender = 'male')
        log('Star loaded completely.')
        return star
    
    def disp_generator(self, pose, shape):
        N = pose.shape[0]
        trans = np.zeros((N, 3))
        base_vertices = STAR.star_forward(self.star, pose, shape, trans)
        new_shape = shape + np.random.normal(0.0, 1.0, 10)

        new_vertices = STAR.star_forward(self.star, pose, new_shape, trans)
        disp = np.subtract(new_vertices, base_vertices)

        return new_shape, disp

    @staticmethod
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

        # vertices = star(pose.to('cuda'), shape.to('cuda'), trans.to('cuda'), disps).to('cpu')
        # kyoosik 0530
        vertices = star(pose.to('cuda'), torch.zeros([1, 10]).to('cuda'), torch.zeros([1, 3]).to('cuda'), torch.zeros([1, 6890, 3]).to('cuda')).to('cpu')
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
