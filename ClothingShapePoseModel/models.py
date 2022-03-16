import numpy as np
import os
import json
from sklearn.decomposition import PCA
import time
from utils import load_obj, save_obj, log
from mlp import MLP, train, get_optimizer
import torch

''' pose shape -> cloth vertices '''
class ClothingShapePoseModelMLP:
    def __init__(self, cloth_name, base_body_dir, epochs = 20000, dims = [2**10, 2**10, 2**10, 2**10, 2**10], train = True, save_path = './model/model1.pt', lr=1e-3):
        '''
            cloth_name: name of cloth.
            base_body_dir: directory of dataset.
            epochs: training epochs
            dims: dimensions of hidden layers of MLP
            save_path: model save path
        '''
        log('Initializing clothing shape model...')

        self.cloth_name = cloth_name
        self.base_body_dir = base_body_dir
        self.epochs = epochs
        self.dims = dims
        self.save_path = save_path
        self.lr = lr
        if train:
            self.get_params_list()
            self.MLP_training()
            torch.save(self.mlp.state_dict(), self.save_path)
        else:
            self.get_params_list(1, 1)
            self.MLP_load()

    def get_params_list(self, N_pose = -1, N_shape = -1):
        '''
            params: dict
            params.keys(): shape, pose, trans, body_vs, cloth_vs
                shape: [N, 10]
                pose: [N, 72]
                trans: [N, 3]
                body_vs: [N, 6890, 3]
                cloth_vs: [N, C, 3]
        '''
        log('Get parameters list...')
        pose_list = os.listdir(self.base_body_dir)
        self.pose_list = [ _ for _ in pose_list if _.startswith('pose_') ]
        if N_pose != -1:
            self.pose_list = self.pose_list[:N_pose]
        log('Length of pose: %d'%len(self.pose_list))
        self.base_body_list = []
        for pose in self.pose_list:
            base_body_list = os.listdir(os.path.join(self.base_body_dir, pose))
            base_body_list = [os.path.join(self.base_body_dir, pose, dir_) for dir_ in base_body_list if dir_.endswith('_params.json')]
            if N_shape != -1:
                base_body_list = base_body_list[:N_shape]
            self.base_body_list += base_body_list
        log('Length of shape: %d'%(len(self.base_body_list) // len(self.pose_list)))
        log('Length of data: %d'%len(self.base_body_list))

        self.params = {'shape':[], 'pose':[], 'trans':[], 'body_vs':[], 'cloth_vs':[]}
        self.textures = None
        self.faces_vs = None
        self.faces_textures = None
        for i, base_body in enumerate(self.base_body_list):
            with open(base_body, "r") as json_:
                param = json.load(json_)
                self.params['shape'].append(param['shape'])
                self.params['pose'].append(param['pose'])
                self.params['trans'].append(param['trans'])
            self.params['body_vs'].append(load_obj(base_body.replace('_params.json', '.obj'), silent = True, verts_only = True)['verts'])
            self.params['cloth_vs'].append(load_obj(os.path.join(base_body.replace('_params.json', ''), self.cloth_name,
                                                                 base_body.replace('_params.json', '').split('/')[-1] + '_%s-scale10.obj'%self.cloth_name), silent = True, verts_only = True)['verts'])
            if i == 0:
                obj_dict = load_obj(os.path.join(base_body.replace('_params.json', ''), self.cloth_name, base_body.replace('_params.json', '').split('/')[-1] + '_%s-scale10.obj'%self.cloth_name), silent = True)
                self.fVerts = obj_dict['fVerts'] # faces of vertex
                self.texture_coords = obj_dict['texture_coords']
                self.fTexture_coords = obj_dict['fTexture_coords']
            if (i+1) % 100 == 0:
                print('\r>>> %d/%d loading...'%(i+1, len(self.base_body_list)), end ='')
        print()
        for key in self.params.keys():
            self.params[key] = np.asarray(self.params[key])

    def load_training_data(self):
        x_shape = torch.FloatTensor(self.params['shape'].reshape([len(self.params['shape']), -1])) # [N, 10]
        x_pose = torch.FloatTensor(self.params['pose'].reshape([len(self.params['pose']), -1])) # [N, 72]
        x = torch.cat((x_shape, x_pose), dim = 1) # [N, 10 + 72]
        y = torch.FloatTensor(self.params['cloth_vs'].reshape([len(self.params['cloth_vs']), -1])) # [N, N_V]
        return x,y

    def MLP_load(self):
        x,y = self.load_training_data()
        dims = [len(x[0])] + self.dims + [len(y[0])]
        log('MLP dimensions: %s.'%dims)
        self.mlp = MLP(dims = dims)
        self.mlp.load_state_dict(torch.load(self.save_path))
        self.mlp.eval()

    def MLP_training(self):
        x,y = self.load_training_data()
        dims = [len(x[0])] + self.dims + [len(y[0])]
        log('MLP dimensions: %s.'%dims)
        self.mlp = MLP(dims = dims)
        optimizer = get_optimizer(self.mlp, lr = self.lr)
        train(self.mlp, optimizer, x, y, self.epochs)

    def drape(self, pose, shape, name = 'sample', save_dir = './example'):
        '''
            pose: [72] (numpy)
            shape: [10] (numpy)
        '''
        log('Draping...')
        start_time = time.time()
        x_shape = torch.FloatTensor(shape.reshape([1, -1]))
        x_pose = torch.FloatTensor(pose.reshape([1, -1]))
        x = torch.cat((x_shape, x_pose), dim = 1)
        self.cloth = self.mlp(x)
        self.cloth = self.cloth.detach().numpy().reshape([-1,3])
        end_time = time.time()
        log('It takes %.03f ms for draping'%( (end_time-start_time) * 1000) )
        obj_dict = {
            'verts': self.cloth,
            'fVerts': self.fVerts,
            'texture_coords': self.texture_coords,
            'fTexture_coords': self.fTexture_coords,
        }
        os.makedirs(save_dir, exist_ok=True)
        save_obj(obj_dict, os.path.join(save_dir, name + '.obj'))

''' DEPRECATED BELOW. I JUST LEFT IT AS A LEGACY '''

''' Basic clothing shape model as proposed in the paper '''
class ClothingShapeModel:
    def __init__(self, pose, clothing_type, cloth_name, base_body_dir, regress_from_vs = False):
        '''
            pose: [72] (numpy)
        '''
        log('Initializing clothing shape model...')
        assert type(pose) == np.ndarray and pose.shape == (72,), 'Pose parameter error.'

        self.regress_from_vs = regress_from_vs
        self.pose = pose
        self.clothing_type = clothing_type
        self.cloth_name = cloth_name
        self.base_body_dir = base_body_dir
        self.get_params_list()
        self.get_pca(n_components = 10)
        self.get_L2_W()


    def get_params_list(self):
        log('Get parameters list...')
        base_body_list = os.listdir(self.base_body_dir)
        self.base_body_list = [dir_ for dir_ in base_body_list if dir_.endswith('_params.json')]
        self.params = {'shape':[], 'pose':[], 'trans':[], 'body_vs':[], 'cloth_vs':[]}
        self.textures = None
        self.faces_vs = None
        self.faces_textures = None
        for base_body in self.base_body_list:
            with open(os.path.join(self.base_body_dir, base_body), "r") as json_:
                param = json.load(json_)
                self.params['shape'].append(param['shape'])
                self.params['pose'].append(param['pose'])
                self.params['trans'].append(param['trans'])
            self.params['body_vs'].append(load_obj(os.path.join(self.base_body_dir, base_body.replace('_params.json', '.obj')))[0])
            self.params['cloth_vs'].append(load_obj(os.path.join(self.base_body_dir,  base_body.replace('_params.json', ''), self.cloth_name,
                                                                 base_body.replace('_params.json', '') + '_%s-scale10.obj'%self.cloth_name))[0])
            if self.textures is None:
                _, self.textures, self.faces_vs, self.faces_textures = load_obj(os.path.join(self.base_body_dir,  base_body.replace('_params.json', ''), self.cloth_name, base_body.replace('_params.json', '') + '_%s-scale10.obj'%self.cloth_name))
                _ = None
        for key in self.params.keys():
            self.params[key] = np.asarray(self.params[key])

    def get_pca(self, n_components = 5):
        log('Executing PCA...')
        pca = PCA(n_components = n_components)
        self.pc = pca.fit_transform(self.params['cloth_vs'].reshape([len(self.params['cloth_vs']), -1])) # principal components
        self.pca = pca

    def get_L2_W(self):
        log('Executing L2-regularized least squares...')
        if self.regress_from_vs:
            #A = np.concatenate([self.params['body_vs'].reshape([len(self.params['body_vs']), -1]),
            #                    self.params['body_vs'].reshape([len(self.params['body_vs']), -1]) ** 2,
            #                    np.ones(len(self.params['body_vs'])).reshape([-1,1])], axis=-1)
            A = np.concatenate([self.params['body_vs'].reshape([len(self.params['body_vs']), -1]),
                                np.ones(len(self.params['body_vs'])).reshape([-1,1])], axis=-1)
        else:
            A = np.concatenate([self.params['shape'], self.params['shape'] ** 2, np.ones(len(self.params['shape'])).reshape([-1,1])], axis=-1)
        y = self.pc
        self.W = np.linalg.lstsq(A, y, rcond=None)[0] # [10, n_components]

    def drape(self, shape, body_vs = None, name = 'sample'):
        '''
            shape: [10] (numpy)
            if self.regress_from_vs:
                body_vs: [6890, 3] (numpy)
        '''
        log('Draping...')
        start_time = time.time()
        if self.regress_from_vs:
            #A = np.concatenate([body_vs.flatten(), body_vs.flatten()**2, np.ones(1)]) # [21]
            A = np.concatenate([body_vs.flatten(), np.ones(1)]) # [21]
        else:
            A = np.concatenate([shape, shape**2, np.ones(1)]) # [21]
        phi = np.matmul( A, self.W ) # [n_components]
        self.cloth = self.pca.mean_
        for i in range(len(phi)):
            self.cloth += phi[i] * self.pca.components_[i]
        self.cloth = self.cloth.reshape([-1,3])
        end_time = time.time()
        log('It takes %.03f ms for draping'%( (end_time-start_time) * 1000) )
        save_obj(self.cloth, self.faces_vs, './%s.obj'%name, self.textures, self.faces_textures)

''' Extended clothing shape model that regresses clothing vertices dircetly from human vertices using MLP '''
class ClothingShapeModelMLP:
    def __init__(self, pose, clothing_type, cloth_name, base_body_dir, regress_from_vs = False):
        '''
            pose: [72] (numpy)
        '''
        log('Initializing clothing shape model...')
        assert type(pose) == np.ndarray and pose.shape == (72,), 'Pose parameter error.'

        self.regress_from_vs = regress_from_vs
        self.pose = pose
        self.clothing_type = clothing_type
        self.cloth_name = cloth_name
        self.base_body_dir = base_body_dir
        self.get_params_list()
        self.MLP_training()

    def get_params_list(self):
        '''
            params: dict
            params.keys(): shape, pose, trans, body_vs, cloth_vs
                shape: [N, 10]
                pose: [N, 72]
                trans: [N, 3]
                body_vs: [N, 6890, 3]
                cloth_vs: [N, C, 3]
        '''
        log('Get parameters list...')
        base_body_list = os.listdir(self.base_body_dir)
        self.base_body_list = [dir_ for dir_ in base_body_list if dir_.endswith('_params.json')]
        self.params = {'shape':[], 'pose':[], 'trans':[], 'body_vs':[], 'cloth_vs':[]}
        self.textures = None
        self.faces_vs = None
        self.faces_textures = None
        for base_body in self.base_body_list:
            with open(os.path.join(self.base_body_dir, base_body), "r") as json_:
                param = json.load(json_)
                self.params['shape'].append(param['shape'])
                self.params['pose'].append(param['pose'])
                self.params['trans'].append(param['trans'])
            self.params['body_vs'].append(load_obj(os.path.join(self.base_body_dir, base_body.replace('_params.json', '.obj')))[0])
            self.params['cloth_vs'].append(load_obj(os.path.join(self.base_body_dir,  base_body.replace('_params.json', ''), self.cloth_name,
                                                                 base_body.replace('_params.json', '') + '_%s-scale10.obj'%self.cloth_name))[0])
            if self.textures is None:
                _, self.textures, self.faces_vs, self.faces_textures = load_obj(os.path.join(self.base_body_dir,  base_body.replace('_params.json', ''), self.cloth_name, base_body.replace('_params.json', '') + '_%s-scale10.obj'%self.cloth_name))
                _ = None
        for key in self.params.keys():
            self.params[key] = np.asarray(self.params[key])

    def load_training_data(self):
        if self.regress_from_vs:
            x = torch.FloatTensor(self.params['body_vs'].reshape([len(self.params['body_vs']), -1]))
        else:
            x = torch.FloatTensor(self.params['shape'].reshape([len(self.params['shape']), -1]))
        y = torch.FloatTensor(self.params['cloth_vs'].reshape([len(self.params['cloth_vs']), -1]))
        return x,y

    def MLP_training(self):
        x,y = self.load_training_data()
        dims = [len(x[0]), 2**10, 2**10, 2**10, 2**10, 2**10, len(y[0])]
        self.mlp = MLP(dims = dims)
        optimizer = get_optimizer(self.mlp, lr = 1e-5)
        train(self.mlp, optimizer, x, y, 20000)

    def drape(self, shape, body_vs, name = 'sample'):
        '''
            shape: [10] (numpy)
            body_vs: [6890, 3] (numpy)
        '''
        log('Draping...')
        start_time = time.time()
        if self.regress_from_vs:
            self.cloth = self.mlp(torch.FloatTensor(body_vs.reshape([1, -1])))
        else:
            self.cloth = self.mlp(torch.FloatTensor(shape.reshape([1, -1])))
        self.cloth = self.cloth.detach().numpy().reshape([-1,3])
        end_time = time.time()
        log('It takes %.03f ms for draping'%( (end_time-start_time) * 1000) )
        save_obj(self.cloth, self.faces_vs, './%s.obj'%name, self.textures, self.faces_textures)
