import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import log
import time

class MLP(nn.Module):
    def __init__(self, dims, act_f = F.relu, dropout = 0):
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

def get_optimizer(model, lr=1e-3):
    return optim.Adam( list(model.parameters()), lr=lr)

def get_loss(Y, Pred, cur_epoch, epochs, device = 'cuda', log = False):
    loss = MSE(Y, Pred, len(Y))

    if log:
        print('\rtraining... %04d/%04d: loss: %03.3f'%(cur_epoch, epochs, loss), end='')

    return loss

def train(model, optimizer, X, Y, epochs):
    log('training MLP starts')
    start_time = time.time()
    model = model.cuda()
    model.train()
    for epoch in range(epochs):
        X = X.cuda()
        Y = Y.cuda()
        Pred = model.forward(X)

        loss = get_loss( Y, Pred, epoch+1, epochs, log = (epoch+1) % 100 == 0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print()
    model = model.to('cpu')
    end_time = time.time()
    log('It elapsed %.2f seconds for training'%(end_time-start_time))
    return model
