import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .utils import log
import time
import numpy as np
import sys
sys.path.append('../utils')
from utils.utils import Utils
# yj
from adamp import AdamP
from torch.cuda.amp import autocast

class MLP(nn.Module):
    def __init__(self, dims, act_f = F.relu, dropout = 0.1): # yj # dropout to 0.1
        '''
            dropout: When dropout > 0, dropout will be activated.
        '''
        super().__init__()
        self.act_f = act_f
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if dropout > 0:
            self.dropouts = nn.ModuleList()
            for i in range(len(dims) - 2):
                self.dropouts.append(nn.Dropout(dropout))
        else:
            self.dropouts = None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.dropouts is not None:
                    x = self.dropouts[i](x)
                x = self.act_f(x)
        return x

def MSE(Y, Pred, batch_size):
    '''
    *** INPUT ***
        Y, Pred: (torch.tensor)
        batch_size: int
    *** OUTPUT ***
        scalar
    '''
    return ((Y - Pred) ** 2).sum() / batch_size

def MAE(Y, Pred, batch_size):
    return (torch.abs(Y - Pred)).sum() / batch_size

def HUBER_LOSS(y, y_pred, delta = 1.0):
    huber_mse = 0.5*((y-y_pred)**2).sum()
    huber_mae = delta * (torch.abs(y - y_pred).sum() - 0.5 * delta)
    output = 0
    if (torch.abs(y - y_pred).sum() <= delta):
        output = huber_mse
    else:
        output = huber_mae
    return output

def get_optimizer(model, lr=1e-3):
    # # Deprecated
    # return optim.Adam( sum([list(model.parameters()) for model in models]), lr=lr)
    return AdamP( sum([list(model.parameters()) for model in models]), lr=lr)

def get_loss(Y, Pred, pose_dependent_deformation, cur_epoch, epochs, device = 'cuda', log = False):
    loss = MSE(Y, Pred, len(Y))
    if pose_dependent_deformation is not None:
        reg_loss = MSE(pose_dependent_deformation, pose_dependent_deformation * 0, len(pose_dependent_deformation)) #* 1e-3
    else:
        reg_loss = 0
    # mae loss 
    # loss = MAE(Y, Pred, len(Y))
    # huber loss 
    # loss = HUBER_LOSS(Y, Pred, len(Y))
    if log:
        print('\rtraining... %04d/%04d: loss: %03.3f reg_loss: %03.3f'%(cur_epoch, epochs, loss, reg_loss), end='')

    return loss + reg_loss

def train(model, model_pose, LBS, optimizer, X_, Y_, X_vertices_, X_pose_, epochs, augmentizer = None, self = None, multi_pose = False):
    log('training MLP starts')
    start_time = time.time()
    model = model.cuda()
    model.train()
    best_eval_loss = 1e10
    
    if multi_pose:
        model_pose = model_pose.cuda()
        model_pose.train()
        self.mean_ = torch.FloatTensor(self.mean_).cuda()
        self.scale_ = torch.FloatTensor(self.scale_).cuda()

    for epoch in range(epochs):
        idxs = np.arange(len(X_))
        np.random.shuffle(idxs)
        X_ = X_[idxs]
        Y_ = Y_[idxs]
        X_vertices_ = X_vertices_[idxs]
        X_pose_ = X_pose_[idxs]
        #print(epoch)
        for t in range(len(X_) // self.batch_size):
            #print('  %d'%t)
            X = X_[t * self.batch_size: (t+1) * self.batch_size].cuda()
            Y = Y_[t * self.batch_size: (t+1) * self.batch_size].cuda()
            X_vertices = X_vertices_[t * self.batch_size: (t+1) * self.batch_size].cuda()
            X_pose = X_pose_[t * self.batch_size: (t+1) * self.batch_size].cuda()
            if augmentizer is not None:
                x_shape = X[:, :10]
                x_pose = X[:, 10:10+63]
                x_shape, disp = self.disp_generator(x_pose, x_shape)
                x = torch.cat((x_shape, x_pose), dim = 1) # [N, 10 + 72]
                y = torch.FloatTensor(self.params['cloth_vs'].reshape([len(self.params['cloth_vs']), -1])) # [N, N_V]
                x = self.encoding(x)
                disp = self.disp_process(disp)
                disp = self.encoding(disp)
                X = torch.cat((x, disp), dim = 1).detach()
            
            with autocast():
                Pred = model.forward(X)
                # 이 부분이 gpu 메모리를 너무 많이 먹음... batch size 를 2000 이상에서 50으로 낮춰야 돌아감
                if multi_pose:
                    pose_dependent_deformation = model_pose.forward(X_pose)

                    Pred = Pred + pose_dependent_deformation
                    for i in range(len(Pred) // self.mini_batch):
                        start_i = i * self.mini_batch
                        end_i = min((i+1) * self.mini_batch, len(Pred))
                        if end_i - start_i != self.mini_batch:
                            start_i = end_i - self.mini_batch
                        _, _, _, Pred_ = LBS.drape(X_vertices[start_i: end_i].reshape(len(X_vertices[start_i: end_i]), -1, 3) * 10,
                                                Pred[start_i: end_i].reshape(len(Pred[start_i: end_i]), -1, 3) * self.scale_ + self.mean_, X_pose[start_i: end_i], LBS.zero_trans, save_dir = None)
                        Pred_ = ((Pred_ - self.mean_) / self.scale_).reshape(len(Pred_), -1)
                        Pred[start_i: end_i] = Pred_
                else:
                    pose_dependent_deformation = None
                loss = get_loss( Y, Pred, pose_dependent_deformation, epoch+1, epochs, log = (epoch+1) % 1 == 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if best_eval_loss > loss:
                torch.save(model.state_dict(), f'./model/model.pt')

        if multi_pose:
            Pred = Pred.reshape(len(Pred), -1, 3) * self.scale_ + self.mean_
            Pred = Pred.detach().cpu().numpy()
            Pred = Pred[0].reshape(-1, 3)
            #body = X_vertices[0].detach().cpu().numpy().reshape(-1, 3)
            obj_dict = {
                'verts': Pred,
                'fVerts': self.fVerts,
                'texture_coords': self.texture_coords,
                'fTexture_coords': self.fTexture_coords,
            }
            Utils.OBJ.save_obj(obj_dict, './sample.obj')

            Y = Y.reshape(len(Y), -1, 3) * self.scale_ + self.mean_
            Y = Y.detach().cpu().numpy()
            Y = Y[0].reshape(-1, 3)
            #body = X_vertices[0].detach().cpu().numpy().reshape(-1, 3)
            obj_dict = {
                'verts': Y,
                'fVerts': self.fVerts,
                'texture_coords': self.texture_coords,
                'fTexture_coords': self.fTexture_coords,
            }
            Utils.OBJ.save_obj(obj_dict, './sample2.obj')

    print()
    model = model.to('cpu')
    end_time = time.time()
    log('It elapsed %.2f seconds for training'%(end_time-start_time))
    return model
