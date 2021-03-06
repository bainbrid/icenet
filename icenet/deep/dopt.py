# Deep Learning optimization functions
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import torch
import uproot
import numpy as np
import sklearn
import psutil
from termcolor import colored,cprint

from matplotlib import pyplot as plt
import pdb
from tqdm import tqdm, trange

import torch.optim as optim
from   torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from icenet.tools import aux
from icenet.tools import io


init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0.0, std=1.0),  # bias terms
    2: lambda x: torch.nn.init.xavier_normal_(x,   gain=1.0),  # weight terms
    3: lambda x: torch.nn.init.xavier_uniform_(x,  gain=1.0),  # conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x,  gain=1.0),  # conv2D filter
    "default": lambda x: torch.nn.init.constant(x, 1.0),       # others
}


def weights_init_all(model, init_funcs):
    """
    Examples:
        model = MyNet()
        weights_init_all(model, init_funcs)
    """
    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)


def weights_init_uniform_rule(m):
    """ Initializes module weights from uniform [-a,a]
    """
    classname = m.__class__.__name__

    # Linear layers
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    """ Initializes module weights from normal distribution
    with a rule sigma ~ 1/sqrt(n)
    """
    classname = m.__class__.__name__
    
    # Linear layers
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        m.bias.data.fill_(0)


def multiclass_cross_entropy(phat, y, N_classes, weights, EPS = 1e-15) :
    """ Per instance weighted cross entropy loss
    (negative log-likelihood)
    """
    
    y = F.one_hot(y, N_classes)

    # Protection
    loss = -y*torch.log(phat + EPS) * weights
    loss = loss.sum() / y.shape[0]

    return loss


def multiclass_focal_entropy(phat, y, N_classes, weights, gamma, EPS = 1e-15) :
    """ Per instance weighted 'focal entropy loss'
    https://arxiv.org/pdf/1708.02002.pdf

    """

    y = F.one_hot(y, N_classes)
    loss = -y * torch.pow(1 - phat, gamma) * torch.log(phat + EPS) * weights
    loss = loss.sum() / y.shape[0]
    
    return loss


def log_sum_exp(x):
    """ 
    http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    """

    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, W):
        """ Initialization """
        self.X = X
        self.Y = Y
        self.W = W

    def __len__(self):
        """ Return the total number of samples """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """

        # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices
        return self.X[index,...], self.Y[index], self.W[index,:]


class DualDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, W):
        """ Initialization """
        self.x = X['x'] # e.g. image tensors
        self.u = X['u'] # e.g. global features 
        
        self.Y = Y
        self.W = W
    
    def __len__(self):
        """ Return the total number of samples """
        return self.x.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices

        O      = {}
        O['x'] = self.x[index,...]
        O['u'] = self.u[index,...]

        return O, self.Y[index], self.W[index,:]


def model_to_cuda(model, device_type='auto'):
    """ Wrapper function to handle CPU/GPU setup
    """
    GPU_chosen = False

    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            GPU_chosen = True
        else:
            device = torch.device('cpu:0')
    else:
        device = param['device']

    model = model.to(device, non_blocking=True)

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(__name__ + f'.model_to_cuda: Multi-GPU {torch.cuda.device_count()}')
        model = nn.DataParallel(model)

    print(__name__ + f'.model_to_cuda: Computing device <{device}> chosen')
    
    if GPU_chosen:
        used  = io.get_gpu_memory_map()[0]
        total = io.torch_cuda_total_memory(device)
        cprint(__name__ + f'.model_to_cuda: device <{device}> VRAM in use: {used:0.2f} / {total:0.2f} GB', 'yellow')
        print('')

    return model, device


def train(model, X_trn, Y_trn, X_val, Y_val, trn_weights, param) :
    """
    Main training loop
    """    
    cprint(__name__ + f""".train: Process RAM usage: {io.process_memory_use():0.2f} GB 
        [total RAM in use {psutil.virtual_memory()[2]} %]""", 'red')
    
    model, device = model_to_cuda(model, param['device'])
    
    # Input checks
    # more than 1 class sample required, will crash otherwise
    if len(np.unique(Y_trn.detach().cpu().numpy())) <= 1:
        raise Exception(__name__ + '.train: Number of classes in ''Y_trn'' <= 1')
    if len(np.unique(Y_val.detach().cpu().numpy())) <= 1:
        raise Exception(__name__ + '.train: Number of classes in ''Y_val'' <= 1')

    # --------------------------------------------------------------------

    if type(X_trn) is dict:
        print(__name__ + f".train: Training samples = {X_trn['x'].shape[0]}, Validation samples = {X_val['x'].shape[0]}")
    else:
        print(__name__ + f'.train: Training samples = {X_trn.shape[0]}, Validation samples = {X_val.shape[0]}')


    # Prints the weights and biases
    print(model)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    cprint(__name__ + f'.train: Number of free parameters = {params} (requires_grad)', 'yellow')
    
    # --------------------------------------------------------------------
    ### Weight initialization
    #print(__name__ + f'.train: Weight initialization')
    #model.apply(weights_init_normal_rule)
    # --------------------------------------------------------------------

    print('')
    
    # Class fractions
    YY = Y_trn.cpu().numpy()
    frac = []
    for i in range(model.C):
        frac.append( sum(YY == i) / YY.size )

    print(__name__ + '.train: Class fractions in the training sample: ')

    ## Classes
    for i in range(len(frac)):
        print(f' {i:4d} : {frac[i]:5.6f} ({sum(YY == i)} counts)')
    print(__name__ + f'.train: Found {len(np.unique(YY))} / {model.C} classes in the training sample')
    

    # Define the optimizer
    if   param['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    elif param['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),  lr=param['learning_rate'], weight_decay=param['weight_decay'])
    elif param['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),   lr=param['learning_rate'], weight_decay=param['weight_decay'])
    else:
        raise Exception(__name__ + f'.train: Unknown optimizer {param["optimizer"]} (use "Adam", "AdamW" or "SGD")')


    # List to store losses
    losses   = []
    trn_aucs = []
    val_aucs = []
    
    print(__name__ + '.train: Training loop ...')

    # Change the shape
    trn_one_hot_weights = np.zeros((len(trn_weights), model.C))
    for i in range(model.C):
        trn_one_hot_weights[YY == i, i] = trn_weights[YY == i]

    params = {'batch_size': param['batch_size'],
            'shuffle'     : True,
            'num_workers' : param['num_workers'],
            'pin_memory'  : True}

    val_one_hot_weights  = np.ones((len(Y_val), model.C))


    ### Generators
    if type(X_trn) is dict:
        training_set   = DualDataset(X_trn, Y_trn, trn_one_hot_weights)
        validation_set = DualDataset(X_val, Y_val, val_one_hot_weights)
    else:
        training_set   = Dataset(X_trn, Y_trn, trn_one_hot_weights)
        validation_set = Dataset(X_val, Y_val, val_one_hot_weights)

    training_generator   = torch.utils.data.DataLoader(training_set,   **params)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)


    ### Epoch loop
    for epoch in tqdm(range(param['epochs']), ncols = 60):

        # Minibatch loop
        sumloss = 0
        nbatch  = 0

        for batch_x, batch_y, batch_weights in training_generator:
            
            # ----------------------------------------------------------------
            # Transfer to (GPU) device memory
            if type(batch_x) is dict: # If multiobject type
                for key in batch_x.keys():
                    batch_x[key] = batch_x[key].to(device, non_blocking=True, dtype=torch.float)
            else:
                batch_x   = batch_x.to(device, non_blocking=True, dtype=torch.float)

            batch_y       = batch_y.to(device, non_blocking=True)
            batch_weights = batch_weights.to(device, non_blocking=True)
            # ----------------------------------------------------------------

            # Noise regularization (NOT ACTIVE)
            #if param['noise_reg'] > 0:
            #    noise   = torch.empty(batch_x.shape).normal_(mean=0, std=param['noise_reg']).to(device, dtype=torch.float32, non_blocking=True)
            #    batch_x = batch_x + noise
            
            # Predict probabilities
            phat = model.softpredict(batch_x)

            # Evaluate loss
            loss = 0
            if   param['lossfunc'] == 'cross_entropy':
                loss = multiclass_cross_entropy(phat, batch_y, model.C, batch_weights)
            elif param['lossfunc'] == 'focal_entropy':
                loss = multiclass_focal_entropy(phat, batch_y, model.C, batch_weights, param['gamma'])
            elif param['lossfunc'] == 'inverse_focal':
                loss = multiclass_inverse_focal(phat, batch_y, model.C, batch_weights, param['gamma'])
            else:
                print(__name__ + '.train: Error with unknown lossfunc ')

            # ------------------------------------------------------------
            optimizer.zero_grad() # Zero gradients
            loss.backward()       # Compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clip gradient for NaN problems

            # Update parameters
            optimizer.step()

            ### Save metrics
            sumloss += loss.item() # item() important for performance
            nbatch  += 1

        avgloss = sumloss / nbatch
        losses.append(avgloss)

        # ================================================================
        # TEST AUC PERFORMANCE SPARSILY (SLOW -- IMPROVE PERFORMANCE)
        if (epoch % 5 == 0) :

            SIGNAL_ID    = 1
            class_labels = np.arange(model.C)
            j = 0

            for gen in [training_generator, validation_generator]:

                auc = 0
                k   = 0
                for batch_x, batch_y, batch_weights in gen:
                                    
                    # ----------------------------------------------------------------
                    # Transfer to (GPU) device memory
                    if type(batch_x) is dict: # If multiobject type
                        for key in batch_x.keys():
                            batch_x[key] = batch_x[key].to(device, non_blocking=True, dtype=torch.float)
                    else:
                        batch_x   = batch_x.to(device, non_blocking=True, dtype=torch.float)

                    batch_y       = batch_y.to(device, non_blocking=True)
                    batch_weights = batch_weights.to(device, non_blocking=True)
                    # ----------------------------------------------------------------

                    phat = model.softpredict(batch_x)

                    # 2-class problems
                    if model.C == 2:
                        metric = aux.Metric(y_true = batch_y.detach().cpu().numpy(), y_soft = phat.detach().cpu().numpy()[:, SIGNAL_ID], valrange=[0,1])
                        if metric.auc > 0:
                            auc += metric.auc
                            k += 1

                    # N-class problems
                    else:
                        auc += sklearn.metrics.roc_auc_score(y_true = batch_y.detach().cpu().numpy(), y_score = phat.detach().cpu().numpy(), \
                            average="weighted", multi_class='ovo', labels=class_labels)
                        k += 1

                # Add AUC
                if j == 0:
                    trn_aucs.append(auc / (k + 1E-12))
                else:
                    val_aucs.append(auc / (k + 1E-12))
                j += 1

            print('Epoch = {} : train loss = {:.3f} [trn AUC = {:.3f}, val AUC = {:.3f}]'. format(epoch, avgloss, trn_aucs[-1], val_aucs[-1]))
                            
        # Just print the loss
        else:
            print('Epoch = {} : train loss = {:.3f}'. format(epoch, avgloss)) 

    return model, losses, trn_aucs, val_aucs
