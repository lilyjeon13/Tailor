import numpy as np
import os
import json
from sklearn.decomposition import PCA
import time
from .utils import load_obj, save_obj, log, read_json
from .mlp import MLP, train, get_optimizer
import torch
import torch.optim as optim
import pickle
import numpy as np

import sys
sys.path.append('../utils')
from torchChamfer import TorchChamfer

def get_A_pose(batch_size = 1):
    PI = np.math.pi
    A_pose = np.zeros([batch_size, 72])

    A_pose[:, 1*3+1] = PI / 12 # 왼쪽 hip
    A_pose[:, 2*3+1] = - PI / 12 # 오른쪽 hip

    A_pose[:, 13*3+2] = - PI / 8 # 왼쪽 collar
    A_pose[:, 14*3+2] = PI / 8 # 오른쪽 collar

    A_pose[:, 16*3+2] = - PI / 6 # 왼쪽 shoulder
    A_pose[:, 17*3+2] = PI / 6 # 오른쪽 shoulder

    A_pose[:, 18*3+1] = - PI / 8 * 1.45 # 왼쪽 elbow
    A_pose[:, 19*3+1] = PI / 8 * 1.45 # 오른쪽 elbow

    A_pose = A_pose.reshape([-1, 24, 3])
    A_pose = A_pose[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
    A_pose = A_pose.reshape([-1, 21 * 3])
    A_pose = [[ 9.2395e-04,  2.5758e-01, -1.2299e-04,  7.1045e-04, -2.5782e-01,
          5.7155e-06,  3.7619e-03,  3.3184e-03,  2.5365e-03, -9.1481e-04,
          5.1081e-03,  5.0539e-04, -1.0761e-03, -5.2566e-03, -4.3568e-04,
         -7.8388e-04, -6.7968e-03, -7.4155e-03,  9.1996e-04, -8.4802e-04,
         -2.1157e-03,  1.8737e-03,  1.0340e-03,  1.2237e-03, -4.7967e-03,
          4.1180e-03,  6.2108e-03,  2.7358e-03,  7.1433e-04,  9.0817e-03,
         -4.1884e-03, -4.1839e-05, -1.3885e-02, -1.1172e-03, -3.7580e-03,
         -3.2470e-03,  7.4858e-03,  1.4553e-02, -3.8656e-01,  7.4517e-03,
         -1.6124e-02,  3.8408e-01,  2.8438e-03,  3.0332e-03,  1.6764e-03,
         -2.6966e-02, -1.1811e-02, -5.3306e-01, -1.8976e-02,  2.0252e-02,
          5.3243e-01, -2.9832e-01, -2.2865e-01, -3.7107e-03, -2.9140e-01,
          2.3198e-01,  2.0267e-03, -5.2778e-03, -1.0741e-02,  2.0040e-02,
         -1.5458e-02, -2.4606e-03, -1.2222e-02] for b in range(batch_size)]
    A_pose = torch.FloatTensor(A_pose).cuda()
    return A_pose

''' pose shape -> cloth vertices '''
class ClothingShapePoseModelMLP_v2:
    def __init__(self, cloth_name, base_body_dir, epochs0 = 800, epochs1 = 20000, epochs2 = 200, dims = [2**10, 2**10, 2**10, 2**10, 2**10], enc_dims = [1024, 1024, 1024], train = True, save_path = './model/model1.pt', lr=1e-3, valid_list = None, HPM = None, augment = True):
        '''
            cloth_name: name of cloth.
            base_body_dir: directory of dataset.
            epochs: training epochs
            dims: dimensions of hidden layers of MLP
            save_path: model save path
            HPM: human parametric model (STAR or SMPLX)
        '''
        log('Initializing clothing shape model...')

        self.cloth_name = cloth_name
        self.base_body_dir = base_body_dir
        self.epochs0 = epochs0
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.dims = dims
        self.enc_dims = enc_dims
        self.save_path = save_path
        self.lr = lr
        self.mean_ = 0
        self.scale_ = 1
        self.disp_scale = 1e+3
        self.valid_list = valid_list
        self.HPM = HPM
        self.augment = augment
        self.n_components = 100 # number of components in PCA for displacements
        self.mini_batch = 50 # for LBS

        if train:
            self.HPM = HPM(batch_size = 1, num_betas = 300 )
            self.get_body_mean(self.HPM) # body mean 은 어차피 zero shape의 mean을 구하기 때문에 저장할 필요 없다.
            self.LBS = ClothingPoseModelMLP(self.cloth_name, self.base_body_dir, HPM, valid_list = self.valid_list, batch_size = self.mini_batch)

            # Phase 0: train auto-encoder of smplx body (A-shape)
            if self.epochs0 > 0:
                self.get_params_list(1,1)
                self.AE_training(HPM)
                torch.save(self.enc.state_dict(), self.save_path.replace('.pt', '_enc.pt'))
            else:
                self.encoder_load()
                self.enc.train()
                self.enc = self.enc.cuda()

            # Phase 1: train clothing model for A-pose.
            if self.epochs1 > 0:
                self.get_params_list(N_pose=1)
                self.batch_size = len(self.params['shape']) # for training
                self.HPM = HPM(batch_size = self.batch_size, num_betas = 300 )

                self.epochs = epochs1
                x, y, x_vertices, x_pose = self.load_training_data()
                dims = [len(x[0])] + self.dims + [len(y[0])]
                log('MLP dimensions: %s.'%dims)
                self.mlp = MLP(dims = dims)
                self.mlp_pose = None

                self.MLP_training(x, y, x_vertices, x_pose, multi_pose = False)

                torch.save(self.mlp.state_dict(), self.save_path)
                np.savez(self.save_path.replace('pt', 'npz'), mean_ = self.mean_, scale_ = self.scale_)
            else:
                self.get_params_list(1,1)
                self.mean_, self.scale_ = self.get_mean_std()
                self.MLP_load()
                self.mlp.train()
                self.mlp = self.mlp.cuda()

            # Phase 2: train clothing pose model for multi-pose.
            if self.epochs2 > 0:
                self.get_params_list(N_shape = 20)
                self.batch_size = self.mini_batch # for training
                self.HPM = HPM(batch_size = self.batch_size, num_betas = 300 )

                self.epochs = epochs2
                x, y, x_vertices, x_pose = self.load_training_data()
                self.mlp_pose = MLP(dims = [63, 1024, 2048, 4096, 9012, len(y[0])])

                self.MLP_training(x, y, x_vertices, x_pose, multi_pose = True)
                torch.save(self.mlp_pose.state_dict(), self.save_path.replace('.pt', '_pose.pt'))

        else:
            self.mean_, self.scale_ = self.get_mean_std()
            self.get_params_list(1, 1)
            self.HPM = HPM(batch_size = 1, num_betas = 300)
            self.get_body_mean(self.HPM)
            self.encoder_load()
            self.MLP_load()
            self.MLP_pose_load()

    def get_body_mean(self, HPM):
        batch_size = HPM.smplx.batch_size #len(self.params['shape'])

        PI = np.math.pi
        A_pose = np.zeros([1, 72])

        A_pose[:, 1*3+1] = PI / 12 # 왼쪽 hip
        A_pose[:, 2*3+1] = - PI / 12 # 오른쪽 hip

        A_pose[:, 13*3+2] = - PI / 8 # 왼쪽 collar
        A_pose[:, 14*3+2] = PI / 8 # 오른쪽 collar

        A_pose[:, 16*3+2] = - PI / 6 # 왼쪽 shoulder
        A_pose[:, 17*3+2] = PI / 6 # 오른쪽 shoulder

        A_pose[:, 18*3+1] = - PI / 8 # 왼쪽 elbow
        A_pose[:, 19*3+1] = PI / 8 # 오른쪽 elbow

        A_pose = A_pose.reshape([-1, 24, 3])
        A_pose = A_pose[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
        A_pose = A_pose.reshape([-1, 21 * 3])
        A_pose = torch.FloatTensor(A_pose).cuda()

        shape = torch.normal(mean=torch.zeros(batch_size,300), std=torch.ones(batch_size,300) * 1 ).cuda() * 0
        pose = torch.normal(mean=torch.ones(batch_size,63), std=torch.zeros(batch_size,63)).cuda() * A_pose
        trans = torch.zeros(batch_size, 3).cuda()
        vertices = HPM.forward(pose, shape, trans)[:1]
        vertices[:, :, 1] -= vertices[:, :, 1].min(axis=1, keepdims = True).values
        self.body_mean = vertices.reshape([1,-1]).detach().cuda() # [1, 10475 * 3]

    def AE_training(self, HPM_, batch_size = 2048, std = 1):
        epochs = self.epochs0
        HPM = HPM_(batch_size = batch_size, num_betas = 300 )

        PI = np.math.pi
        A_pose = np.zeros([1, 72])

        A_pose[:, 1*3+1] = PI / 12 # 왼쪽 hip
        A_pose[:, 2*3+1] = - PI / 12 # 오른쪽 hip

        A_pose[:, 13*3+2] = - PI / 8 # 왼쪽 collar
        A_pose[:, 14*3+2] = PI / 8 # 오른쪽 collar

        A_pose[:, 16*3+2] = - PI / 6 # 왼쪽 shoulder
        A_pose[:, 17*3+2] = PI / 6 # 오른쪽 shoulder

        A_pose[:, 18*3+1] = - PI / 8 # 왼쪽 elbow
        A_pose[:, 19*3+1] = PI / 8 # 오른쪽 elbow

        A_pose = A_pose.reshape([-1, 24, 3])
        A_pose = A_pose[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
        A_pose = A_pose.reshape([-1, 21 * 3])
        A_pose = torch.FloatTensor(A_pose).cuda()

        enc_dims = self.enc_dims #[10475*3, 10475, 4096, 1024]
        dec_dims = self.enc_dims[::-1] #[1024, 4096, 10475, 10475*3]

        self.enc = MLP(dims = enc_dims).cuda()
        self.dec = MLP(dims = dec_dims).cuda()
        optimizer = optim.Adam( list(self.enc.parameters()) + list(self.dec.parameters()), lr=1e-4)
        for e in range(epochs):
            shape = torch.normal(mean=torch.zeros(batch_size,300), std=torch.ones(batch_size,300) * std ).cuda()
            pose = torch.normal(mean=torch.ones(batch_size,63), std=torch.zeros(batch_size,63)).cuda() * A_pose
            trans = torch.zeros(batch_size, 3).cuda()
            vertices = HPM.forward(pose, shape, trans)
            vertices[:, :, 1] -= vertices[:, :, 1].min(axis=1, keepdims = True).values
            vertices = vertices.reshape([batch_size, -1]) - self.body_mean # * 10

            pred = self.dec.forward(self.enc.forward(vertices))
            loss = ((pred - vertices) ** 2).sum() / batch_size# * 1e-3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vertices += self.body_mean
            pred += self.body_mean

            if (e+1) % 10 == 0 or e+1 == epochs:
                HPM_.save_smplx_obj(vertices[0].reshape([-1,3]).detach().cpu().numpy().tolist(), './sample.obj')
                HPM_.save_smplx_obj(self.params['body_vs'][0].reshape([-1,3]), './sample2.obj')
                HPM_.save_smplx_obj(pred[0].reshape([-1,3]).detach().cpu().numpy().tolist(), './sample3.obj')

                print('\r>>> epoch %d/%d Auto-encoder loss: %.4f'%(e+1, epochs, loss.item()), end ='')
        print()
        #self.enc = self.enc.cpu()
        self.dec = self.dec.cpu()

    def disp_process(self, disp, w_pca = True):
        disp = disp * self.disp_scale
        #disp = disp[:, ::24]
        if w_pca:
            disp = disp.reshape(len(disp), -1) # [batch, N_vertices * 3]
            disp = disp.cpu().detach().numpy()
            disp = self.pca.transform(disp)
            disp = torch.FloatTensor(disp).cuda() # [batch, n_components]
        return disp

    def disp_generator(self, pose, shape, std = 2):
        N = pose.shape[0]
        trans = torch.zeros((N, 3)).cuda()
        base_vertices = self.HPM.forward(pose * 0, shape, trans)
        delta_shape = torch.normal(mean=torch.zeros(N,10), std=torch.ones(N,10) * std ).cuda() * torch.normal(mean=torch.zeros(N,1), std=torch.ones(N,1)).cuda()
        delta_shape[:, 0] = 0
        new_shape = shape + delta_shape

        new_vertices = self.HPM.forward(pose * 0, new_shape, trans)
        disp = base_vertices - new_vertices

        return new_shape, disp.reshape([N, -1])

    def get_mean_std(self):
        mean_std = np.load(self.save_path.replace('pt', 'npz'))
        mean_ = mean_std['mean_']
        scale_ = mean_std['scale_']
        return mean_, scale_

    def get_params_list(self, N_pose = -1, N_shape = -1, normalize = True):
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
        self.pose_list.sort()
        #self.pose_list = self.pose_list[1:]
        if N_pose != -1:
            self.pose_list = self.pose_list[:N_pose]
        log('Length of pose: %d'%len(self.pose_list))
        self.base_body_list = []
        for pose in self.pose_list:
            base_body_list = os.listdir(os.path.join(self.base_body_dir, pose))
            base_body_list = [os.path.join(self.base_body_dir, pose, dir_) for dir_ in base_body_list if (dir_.endswith('_params.json') and '0000_0' not in dir_)]  # 각 pose 맨 처음 subject가  T-pose 에서 해당 pose 로 서서히 변해가는 과정을 담은 obj 를 제외
            base_body_list.sort()
            if N_shape != -1:
                base_body_list = base_body_list[:N_shape]
            self.base_body_list += base_body_list
        log('Length of shape: %d'%(len(self.base_body_list) // len(self.pose_list)))
        log('Length of data: %d'%len(self.base_body_list))

        # valid list 을 Check is clothing valid 를 통해 생성
        #valid_data_list = read_json(os.path.join(self.base_body_dir, '%s_valid_list.json'%self.cloth_name))['path']
        if self.valid_list is not None:
            base_body_list = []
            for b in self.base_body_list:
                pose = int(b[b.find('pose_') + 5:b.find('pose_') + 9])
                subject = int(b[b.find('subject_') + 8:b.find('subject_') + 12])
                #if pose >0 : continue
                for target in self.valid_list:
                    pose_target = int(target[target.find('pose_') + 5:target.find('pose_') + 9])
                    subject_target = int(target[target.find('subject_') + 8:target.find('subject_') + 12])
                    if pose == pose_target and subject == subject_target:
                        base_body_list.append(b)
                        break

        log('Valid data length: %d'%len(base_body_list))
        self.base_body_list = base_body_list

        self.params = {'shape':[], 'pose':[], 'trans':[], 'body_vs':[], 'body_vs_A':[], 'cloth_vs':[]}
        self.textures = None
        self.faces_vs = None
        self.faces_textures = None
        for i, base_body in enumerate(self.base_body_list):
            with open(base_body, "r") as json_:
                param = json.load(json_)
                self.params['shape'].append(param['shape'])
                self.params['pose'].append(param['pose'])
                self.params['trans'].append(param['trans'])
            self.params['body_vs_A'].append(load_obj(base_body.replace('_params.json', '_A.obj'), silent = True, verts_only = True)['verts'])
            self.params['body_vs'].append(load_obj(base_body.replace('_params.json', '.obj'), silent = True, verts_only = True)['verts'])
            self.params['cloth_vs'].append(np.asarray(load_obj(os.path.join(base_body.replace('_params.json', ''), self.cloth_name,
                                                                 base_body.replace('_params.json', '').split('/')[-1] + '_%s-scale10.obj'%self.cloth_name), silent = True, verts_only = True)['verts']))
            self.params['cloth_vs'][-1][np.where(self.params['cloth_vs'][-1][:, 1] < -10)[0]] *= 0 # 가끔씩 vertex 가 옷으로부터 떨어져서 한없이 -y 로 떨어지는 경우가 있다. 이를 예방한다.
            if i == 0:
                obj_dict = load_obj(os.path.join(base_body.replace('_params.json', ''), self.cloth_name, base_body.replace('_params.json', '').split('/')[-1] + '_%s-scale10.obj'%self.cloth_name), silent = True)
                self.fVerts = obj_dict['fVerts'] # faces of vertex
                self.texture_coords = obj_dict['texture_coords']
                self.fTexture_coords = obj_dict['fTexture_coords']
            if (i+1) % 100 == 0:
                print('\r>>> %d/%d loading...'%(i+1, len(self.base_body_list)), end ='')
        print()
        if normalize and type(self.mean_) == type(0) and self.mean_ == 0:
            self.mean_ = np.mean(self.params['cloth_vs'], axis = (0,1))
            self.scale_ = np.std(self.params['cloth_vs'], axis = (0,1))
        log('mean: %s, scale_:%s'%(self.mean_, self.scale_))
        self.params['cloth_vs'] = np.asarray(self.params['cloth_vs'])
        self.params['cloth_vs'] -= self.mean_
        self.params['cloth_vs'] /= self.scale_

        for key in self.params.keys():
            self.params[key] = np.asarray(self.params[key])

    def encoding(self, X, N = 10):
        X_new = []
        X_new.append(X)
        for i in range(1, N + 1):
            #X_new.append(torch.sin(X*i))
            X_new.append(torch.sin(X/i))
            #X_new.append(torch.cos(X*i))
            X_new.append(torch.cos(X/i))
        return torch.cat(X_new, axis=1)

    def train_disp_PCA(self, n_components = 10, disp_n = 10):
        log('Train displacement PCA...')
        x_shape = torch.FloatTensor(self.params['shape'].reshape([len(self.params['shape']), -1])).cuda() # [N, 10]
        x_pose = torch.FloatTensor(self.params['pose'].reshape([len(self.params['pose']), -1])).cuda() # [N, 72]
        disps = []
        for i in range(disp_n):
            _, disp = self.disp_generator(x_pose, x_shape) # disp: [batch, N_vertices, 3]
            disp = self.disp_process(disp, w_pca = False)
            disps.append(disp.detach().cpu())
        disp = torch.cat(disps, dim = 0) # [batch * disp_n, N_vertices, 3]
        disp = disp.numpy()

        pca = PCA(n_components = n_components)
        self.pc = pca.fit_transform(disp) # principal components
        self.pca = pca
        with open(self.save_path.replace('pt', 'joblib'), 'wb') as file:
            pickle.dump(self.pca, file)

    def load_training_data(self):
        x_vertices = torch.FloatTensor(self.params['body_vs_A'].reshape([len(self.params['body_vs_A']), -1])) # [N, 10475*3]
        x_pose = torch.FloatTensor(self.params['pose'].reshape([len(self.params['pose']), -1])) # [N, 63]
        x_enc = self.enc(x_vertices.cuda()  - self.body_mean)
        y = torch.FloatTensor(self.params['cloth_vs'].reshape([len(self.params['cloth_vs']), -1])) # [N, N_V]
        return x_enc.detach(), y, x_vertices, x_pose

    def encoder_load(self):
        enc_dims = self.enc_dims
        log('Encoder dimensions: %s.'%enc_dims)
        self.enc = MLP(dims = enc_dims)
        self.enc.load_state_dict(torch.load(self.save_path.replace('.pt', '_enc.pt')))
        self.enc.eval()
        self.enc = self.enc.cuda()

    def MLP_load(self):
        x,y, x_vertices, x_pose = self.load_training_data()
        dims = [len(x[0])] + self.dims + [len(y[0])]
        log('MLP dimensions: %s.'%dims)
        self.mlp = MLP(dims = dims)
        self.mlp.load_state_dict(torch.load(self.save_path))
        self.mlp.eval()

    def MLP_pose_load(self):
        x,y, x_vertices, x_pose = self.load_training_data()
        self.mlp_pose = MLP(dims = [63, 1024, 2048, 4096, 9012, len(y[0])])
        self.mlp_pose.load_state_dict(torch.load(self.save_path.replace('.pt', '_pose.pt')))
        self.mlp_pose.eval()

    def MLP_training(self, x, y, x_vertices, x_pose, multi_pose = False):
        if multi_pose:
            optimizer = optim.Adam( list(self.mlp_pose.parameters()), lr=1e-3)
        else:
            optimizer = optim.Adam( list(self.mlp.parameters()), lr=1e-4)
        if self.augment:
            augmentizer = self.disp_generator
        else:
            augmentizer = None
        train(self.mlp, self.mlp_pose, self.LBS, optimizer, x, y, x_vertices, x_pose, self.epochs, augmentizer = augmentizer, self = self, multi_pose = multi_pose)

    # shape model + pose model 
    def drape(self, vertices, pose, disp, name = 'sample', save_dir = './example'):
        '''
            vertices: [10475, 3] (numpy)
        '''
        if save_dir is not None:
            log('Draping...')
        start_time = time.time()
        x = torch.FloatTensor(vertices.reshape([1, -1])).cuda()
        x_shape = torch.FloatTensor(self.params['shape'].reshape([len(self.params['shape']), -1])).cuda() # [N, 10]
        x_pose = torch.FloatTensor(self.params['pose'].reshape([len(self.params['pose']), -1])).cuda() # [N, 72]
        if self.augment and disp is not None:
            x = torch.cat((x_shape, x_pose), dim = 1) # [N, 10 + 72]
            x = self.encoding(x)
            disp = torch.FloatTensor(disp.reshape([1, -1])).cuda()
            disp = self.disp_process(disp)
            disp = self.encoding(disp)
            x = torch.cat((x, disp), dim = 1)
        elif self.augment:
            x_shape, disp = self.disp_generator(x_pose, x_shape, std = 0)
            x = torch.cat((x_shape, x_pose), dim = 1) # [N, 10 + 72]
            x = self.encoding(x)
            x = torch.cat((x, disp), dim = 1)
        else:
            x = self.enc(x - self.body_mean)
        self.cloth = self.mlp(x.cpu())
        self.pose_dependent_deformation = self.mlp_pose(pose.cpu())
        self.cloth = self.cloth + self.pose_dependent_deformation #* 0
        self.cloth = self.cloth.detach().numpy().reshape([-1,3])
        self.cloth *= self.scale_
        self.cloth += self.mean_
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
        return self.cloth

class ClothingPoseModelMLP:
    def __init__(self, cloth_name, base_body_dir, HPM = None, valid_list = None, batch_size = 1):
        '''
            cloth_name: name of cloth.
            base_body_dir: directory of dataset.
            HPM: human parametric model (STAR or SMPLX)
        '''
        log('Initializing clothing shape model...')

        self.cloth_name = cloth_name
        self.base_body_dir = base_body_dir
        self.mean_ = 0
        self.scale_ = 1
        self.valid_list = valid_list
        self.HPM_ = HPM
        self.HPM = HPM(batch_size = batch_size, num_betas = 300)
        self.can_pose = get_A_pose(batch_size) # [1, 63], torch.FloatTensor, cuda
        self.zero_trans = torch.FloatTensor(np.zeros([batch_size, 3])).cuda()
        self.parents = self.HPM.smplx.parents[1:]
        self.cd = TorchChamfer()
        self.get_params_list(1, 1)

    def get_params_list(self, N_pose = -1, N_shape = -1, normalize = True):
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
        self.pose_list.sort()
        #self.pose_list = self.pose_list[1:]
        if N_pose != -1:
            self.pose_list = self.pose_list[:N_pose]
        log('Length of pose: %d'%len(self.pose_list))
        self.base_body_list = []
        for pose in self.pose_list:
            base_body_list = os.listdir(os.path.join(self.base_body_dir, pose))
            base_body_list = [os.path.join(self.base_body_dir, pose, dir_) for dir_ in base_body_list if (dir_.endswith('_params.json') and '0000_0' not in dir_)]  # 각 pose 맨 처음 subject가  T-pose 에서 해당 pose 로 서서히 변해가는 과정을 담은 obj 를 제외
            base_body_list.sort()
            if N_shape != -1:
                base_body_list = base_body_list[:N_shape]
            self.base_body_list += base_body_list
        log('Length of shape: %d'%(len(self.base_body_list) // len(self.pose_list)))
        log('Length of data: %d'%len(self.base_body_list))

        # valid list 을 Check is clothing valid 를 통해 생성
        #valid_data_list = read_json(os.path.join(self.base_body_dir, '%s_valid_list.json'%self.cloth_name))['path']
        if self.valid_list is not None:
            base_body_list = []
            for b in self.base_body_list:
                pose = int(b[b.find('pose_') + 5:b.find('pose_') + 9])
                subject = int(b[b.find('subject_') + 8:b.find('subject_') + 12])
                if pose >0 : continue
                for target in self.valid_list:
                    pose_target = int(target[target.find('pose_') + 5:target.find('pose_') + 9])
                    subject_target = int(target[target.find('subject_') + 8:target.find('subject_') + 12])
                    if pose == pose_target and subject == subject_target:
                        base_body_list.append(b)
                        break

        log('Valid data length: %d'%len(base_body_list))
        self.base_body_list = base_body_list

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
            self.params['cloth_vs'].append(np.asarray(load_obj(os.path.join(base_body.replace('_params.json', ''), self.cloth_name,
                                                                 base_body.replace('_params.json', '').split('/')[-1] + '_%s-scale10.obj'%self.cloth_name), silent = True, verts_only = True)['verts']))
            self.params['cloth_vs'][-1][np.where(self.params['cloth_vs'][-1][:, 1] < -10)[0]] *= 0 # 가끔씩 vertex 가 옷으로부터 떨어져서 한없이 -y 로 떨어지는 경우가 있다. 이를 예방한다.
            if i == 0:
                obj_dict = load_obj(os.path.join(base_body.replace('_params.json', ''), self.cloth_name, base_body.replace('_params.json', '').split('/')[-1] + '_%s-scale10.obj'%self.cloth_name), silent = True)
                self.fVerts = obj_dict['fVerts'] # faces of vertex
                self.texture_coords = obj_dict['texture_coords']
                self.fTexture_coords = obj_dict['fTexture_coords']
            if (i+1) % 100 == 0:
                print('\r>>> %d/%d loading...'%(i+1, len(self.base_body_list)), end ='')
        print()
        if normalize and type(self.mean_) == type(0):
            self.mean_ = np.mean(self.params['cloth_vs'], axis = (0,1))
            self.scale_ = np.std(self.params['cloth_vs'], axis = (0,1))
        log('mean: %s, scale_:%s'%(self.mean_, self.scale_))
        self.params['cloth_vs'] = np.asarray(self.params['cloth_vs'])
        self.params['cloth_vs'] -= self.mean_
        self.params['cloth_vs'] /= self.scale_

        for key in self.params.keys():
            self.params[key] = np.asarray(self.params[key])

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

    def set_pose(self, can_verts, pose, trans, lbs_weights, device = 'cuda', cloth_vertices = None):
        '''
        *** INPUT ***
            can_verts: [batch_size, N_V, 3]
            pose: [batch_size, 63]
        '''
        assert len(can_verts) == len(pose), "Batch sizes of can_verts and pose should be same."
        body_verts = can_verts
        if cloth_vertices is not None:
            can_verts = cloth_vertices

        batch_size = len(can_verts)
        N_V = can_verts.shape[1]
        N_K = 55
        # pose and can_pose to SMPLX full pose
        pose = torch.cat([self.HPM.smplx.global_orient.reshape(-1, 1, 3),
                               pose.reshape(-1, 21, 3),
                               self.HPM.smplx.jaw_pose.reshape(-1, 1, 3),
                               self.HPM.smplx.leye_pose.reshape(-1, 1, 3),
                               self.HPM.smplx.reye_pose.reshape(-1, 1, 3),
                               torch.einsum('bi,ij->bj', [self.HPM.smplx.left_hand_pose, self.HPM.smplx.left_hand_components]).reshape(-1, 15, 3),
                               torch.einsum('bi,ij->bj', [self.HPM.smplx.right_hand_pose, self.HPM.smplx.right_hand_components]).reshape(-1, 15, 3)],
                              dim=1).reshape(-1, 165)
        can_pose = torch.cat([self.HPM.smplx.global_orient.reshape(-1, 1, 3),
                               self.can_pose.reshape(-1, 21, 3),
                               self.HPM.smplx.jaw_pose.reshape(-1, 1, 3),
                               self.HPM.smplx.leye_pose.reshape(-1, 1, 3),
                               self.HPM.smplx.reye_pose.reshape(-1, 1, 3),
                               torch.einsum('bi,ij->bj', [self.HPM.smplx.left_hand_pose, self.HPM.smplx.left_hand_components]).reshape(-1, 15, 3),
                               torch.einsum('bi,ij->bj', [self.HPM.smplx.right_hand_pose, self.HPM.smplx.right_hand_components]).reshape(-1, 15, 3)],
                              dim=1).reshape(-1, 165)

        C = can_verts.reshape(batch_size, N_V * 3) # [batch_size, N_V * 3]
        R = self.pose2rodrigues(pose.contiguous().view(-1, 3)).view(batch_size, -1, 3, 3) # [batch_size, N_K, 3, 3]
        # T_pose = T + torch.matmul(self.P.T, R[:, 1:].view(-1, self.N_KR).T).view(self.N_V * 3, self.N_S * N_P).T # [N_S * 1, N_V * 3], add pose blend shapes
        C_pose = C
        C_pose = C_pose.view(batch_size, N_V, 3) # [batch_size, N_V, 3]
        C = C.view(batch_size, N_V, 3) # [batch_size, N_V, 3]

        #J = torch.matmul(self.J.T, T.view(self.N_S * N_P, self.N_V * 3).T).T.view(self.N_S, self.N_K, 3) # [N_S * 1, N_K, 3]
        J = torch.einsum('bik,ji->bjk', [body_verts, self.HPM.smplx.J_regressor]).view(batch_size, N_K, 3) # [batch_size, N_K, 3]
        J_ = J.clone() # [batch_size, N_K, 3]
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parents, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1) # [batch_size, N_K, 3, 4]
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, N_K, -1, -1) # [batch_size, N_K, 3, 4]
        G_ = torch.cat([G_, pad_row], dim=2) # [batch_size, N_K, 4, 4]
        G = [G_[:, 0].clone()] # [1, batch_size, 4, 4]
        for i in range(1, N_K):
            G.append(torch.matmul(G[self.parents[i - 1]], G_[:, i, :, :])) # [N_K, batch_size, 4, 4]
        G = torch.stack(G, dim=1) # [batch_size, N_K, 4, 4]
        rest = torch.cat([J, torch.zeros(batch_size, N_K, 1).to(device)], dim=2).view(batch_size, N_K, 4, 1) # [batch_size, N_K, 4] -> [batch_size, N_K, 4, 1]

        zeros = torch.zeros(batch_size, N_K, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1) # [batch_size, N_K, 4, 4]
        rest = torch.matmul(G, rest) # [batch_size, N_K, 4, 4]
        G = G - rest
        #C_ = torch.matmul(lbs_weights, G.permute(1, 0, 2, 3).contiguous().view(N_K, -1)).view(N_V, batch_size, 4,4).transpose(0, 1) # [batch_size, N_V, 4, 4]
        C_ = torch.matmul(lbs_weights, G.permute(1, 0, 2, 3).contiguous().view(N_K, -1)).view(batch_size, N_V, batch_size, 4,4).transpose(1, 2) # [batch_size, batch_size, N_V, 4, 4]
        C_ = torch.cat([C_[i:i+1, i] for i in range(batch_size)]) # [batch_size, N_V, 4, 4]
        rest_shape_h = torch.cat([C_pose, torch.ones_like(C_pose)[:, :, [0]]], dim=-1) # [batch_size, N_V, 4]
        v = torch.matmul(C_, rest_shape_h[:, :, :, None])[:, :, :3, 0] # [batch_size, N_V, 3]
        v = v + trans.view(len(v), 1, 3)
        #self.joint_locs = J + trans.view(len(v), 1, 3) # [N_S, N_K, 3]
        return v

    def get_lbs_weights(self, body_vertices, cloth_vertices, body_lbs_weights, K=10):
        '''
        *** INPUT ***
            body_vertices: [batch_size, N_V, 3]
            cloth_vertices: [batch_size, N_C, 3]
            body_lbs_weights: [N_V, N_K]
        '''
        batch_size = len(body_vertices)
        N_V, N_K = body_lbs_weights.shape[0], body_lbs_weights.shape[1]
        N_C = cloth_vertices.shape[1]
        distances = torch.zeros([batch_size, K, N_C]).cuda() # [batch_size, len(K), N_C]
        lbss = torch.zeros([batch_size, K, N_C, N_K]).cuda() # [batch_size, len(K), N_C, N_K]
        distance_xy, idxs_xy = self.cd.chamfer_distance(cloth_vertices, body_vertices, K=K, return_idxs = True, return_all = True) # [batch_size, N_C, K]

        for i in range(K):
            distances[:, i] = distance_xy[:, :, i]
            lbss[:, i] = body_lbs_weights[idxs_xy[:, :, i].view(-1)].view(batch_size, -1, N_K)
        inverse_distances = 1 / distances
        relative_inverse_distances = inverse_distances / inverse_distances.sum(axis = 1, keepdims = True)
        lbs = (lbss * relative_inverse_distances.view(relative_inverse_distances.shape + (1,))).sum(axis = 1)

        return lbs.detach()
    
    # apply LBS
    def drape(self, body_vertices, cloth_vertices, pose, trans, name = 'sample', save_dir = './example', cloth_lbs_weights = None):
        '''
            vertices: [10475, 3] (numpy)
        '''
        if save_dir is not None:
            log('Draping...')
        batch_size = len(pose)
        start_time = time.time()

        if cloth_lbs_weights is None:
            cloth_lbs_weights = self.get_lbs_weights(body_vertices, cloth_vertices, self.HPM.smplx.lbs_weights)

        body_lbs_weights = self.HPM.smplx.lbs_weights.view(1, -1, 55).tile(batch_size, 1, 1)
        T_vertices = self.set_pose(body_vertices, -self.can_pose, trans, lbs_weights = body_lbs_weights)
        vertices = self.set_pose(T_vertices, pose, trans, lbs_weights = body_lbs_weights)

        T_cloth_vertices = self.set_pose(body_vertices, -self.can_pose, trans, lbs_weights = cloth_lbs_weights, cloth_vertices = cloth_vertices)
        cloth_vertices = self.set_pose(T_vertices, pose, trans, lbs_weights = cloth_lbs_weights, cloth_vertices = T_cloth_vertices)

        end_time = time.time()


        if save_dir is not None:

            log('It takes %.03f ms for draping'%( (end_time-start_time) * 1000) )
            os.makedirs(save_dir, exist_ok=True)
            self.HPM_.save_smplx_obj(T_vertices[0], os.path.join(save_dir, name + '_T.obj'))
            self.HPM_.save_smplx_obj(vertices[0], os.path.join(save_dir, name + '_pose.obj'))

            obj_dict = {
                'verts': T_cloth_vertices[0],
                'fVerts': self.fVerts,
                'texture_coords': self.texture_coords,
                'fTexture_coords': self.fTexture_coords,
            }
            os.makedirs(save_dir, exist_ok=True)
            save_obj(obj_dict, os.path.join(save_dir, name + '_cloth_T.obj'))

            obj_dict = {
                'verts': cloth_vertices[0],
                'fVerts': self.fVerts,
                'texture_coords': self.texture_coords,
                'fTexture_coords': self.fTexture_coords,
            }
            os.makedirs(save_dir, exist_ok=True)
            save_obj(obj_dict, os.path.join(save_dir, name + '_cloth_pose.obj'))
        return T_vertices, T_cloth_vertices, vertices, cloth_vertices
