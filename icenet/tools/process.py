# Common input & data reading routines
#
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

import argparse
import yaml
import numpy as np
from termcolor import colored, cprint
import os
import copy
import sys
import pickle
import torch


import icenet.deep.train as train
import icenet.deep.predict as predict


from icenet.tools import io
from icenet.tools import prints
from icenet.tools import aux
from icenet.tools import reweight
from icenet.tools import plots


import matplotlib.pyplot as plt


# Command line arguments
from glob import glob
from braceexpand import braceexpand


# ******** GLOBALS *********
roc_mstats    = []
roc_labels    = []
# **************************


def read_config(config_path='./configs/xyz'):
    """
    Commandline and YAML configuration reader
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type = str, default='tune0')
    parser.add_argument("--datapath", type = str, default=".")
    parser.add_argument("--datasets", type = str, default="*")

    cli = parser.parse_args()

    # -------------------------------------------------------------------
    ## Read configuration
    args = {}
    config_yaml_file = cli.config + '.yml'
    with open(config_path + '/' + config_yaml_file, 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    args['config'] = cli.config

    # -------------------------------------------------------------------
    ### Set image and graph constructions on/off
    args['graph_on'] = False
    args['image_on'] = False

    for i in range(len(args['active_models'])):
        ID    = args['active_models'][i]
        param = args[f'{ID}_param']

        if ('graph' in param['train']) or ('graph' in param['predict']):
            args['graph_on'] = True
        if ('image' in param['train']) or ('image' in param['predict']):
            args['image_on'] = True

    print('\n')
    cprint(__name__ + f'.read_config: graph_on = {args["graph_on"]}', 'yellow')
    cprint(__name__ + f'.read_config: image_on = {args["image_on"]}', 'yellow')    

    # -------------------------------------------------------------------
    # Do brace expansion
    datasets = list(braceexpand(cli.datasets))

    # Parse input files into a list
    args['root_files'] = list()
    for data in datasets:
        filepath = glob(cli.datapath + '/' + data + '.root')
        if filepath != []:
            for i in range(len(filepath)):
                args['root_files'].append(filepath[i])

    if args['root_files'] == []:
        args['root_files'] = [cli.datapath]

    # -------------------------------------------------------------------
    
    print(args)
    print('')
    print(" torch.__version__: " + torch.__version__)
    print("")
    print(" Try 'filename_*' ")
    print(" Try 'filename_[0-99]' ")
    print(" Try 'filename_0' ")
    print(" Try 'filename_{0,3,4}' ")
    print(" Google <glob wildcards> and brace expansion.")
    print("")

    return args, cli


def train_models(data, data_tensor=None, data_kin=None, data_graph=None, trn_weights=None, args=None) :
    """
    Train ML/AI models.
    """
    
    print(__name__ + f": Input with {data.trn.x.shape[0]} events and {data.trn.x.shape[1]} dimensions ")

    # @@ Tensor normalization @@
    if args['image_on'] and (args['varnorm_tensor'] == 'zscore'):
            
        print('\nZ-score normalizing tensor variables ...')
        X_mu_tensor, X_std_tensor = io.calc_zscore_tensor(data_tensor['trn'])
        for key in ['trn', 'val']:
            data_tensor[key] = io.apply_zscore_tensor(data_tensor[key], X_mu_tensor, X_std_tensor)
        
        # Save it for the evaluation
        pickle.dump([X_mu_tensor, X_std_tensor], open(args["modeldir"] + '/zscore_tensor.dat', 'wb'))    
    
    # --------------------------------------------------------------------

    # @@ Truncate outliers (component by component) from the training set @@
    if args['outlier_param']['algo'] == 'truncate' :
        for j in range(data.trn.x.shape[1]):

            minval = np.percentile(data.trn.x[:,j], args['outlier_param']['qmin'])
            maxval = np.percentile(data.trn.x[:,j], args['outlier_param']['qmax'])

            data.trn.x[data.trn.x[:,j] < minval, j] = minval
            data.trn.x[data.trn.x[:,j] > maxval, j] = maxval

    # @@ Variable normalization @@
    if args['varnorm'] == 'zscore' :

        print('\nZ-score normalizing variables ...')
        X_mu, X_std = io.calc_zscore(data.trn.x)
        data.trn.x  = io.apply_zscore(data.trn.x, X_mu, X_std)
        data.val.x  = io.apply_zscore(data.val.x, X_mu, X_std)

        # Save it for the evaluation
        pickle.dump([X_mu, X_std], open(args['modeldir'] + '/zscore.dat', 'wb'))

    elif args['varnorm'] == 'madscore' :

        print('\nMAD-score normalizing variables ...')
        X_m, X_mad  = io.calc_madscore(data.trn.x)
        data.trn.x  = io.apply_madscore(data.trn.x, X_m, X_mad)
        data.val.x  = io.apply_madscore(data.val.x, X_m, X_mad)

        # Save it for the evaluation
        pickle.dump([X_m, X_mad], open(args['modeldir'] + '/madscore.dat', 'wb'))
    
    prints.print_variables(data.trn.x, data.ids)

    ### Pick training data into PyTorch format
    X_trn = torch.from_numpy(data.trn.x).type(torch.FloatTensor)
    Y_trn = torch.from_numpy(data.trn.y).type(torch.LongTensor)

    X_val = torch.from_numpy(data.val.x).type(torch.FloatTensor)
    Y_val = torch.from_numpy(data.val.y).type(torch.LongTensor)
    
    
    # Loop over active models
    for i in range(len(args['active_models'])):

        ID    = args['active_models'][i]
        param = args[f'{ID}_param']
        print(f'Training <{ID}> | {param} \n')

        if   param['train'] == 'graph':
            
            if args['raytune_param']['active']:
                train.raytune_main(data_trn=data_graph['trn'], data_val=data_graph['val'], args=args, param=param)
            else:
                train.train_graph(data_trn=data_graph['trn'], data_val=data_graph['val'], args=args, param=param)

        elif param['train'] == 'graph_xgb':
            train.train_graph_xgb(data_trn=data_graph['trn'], data_val=data_graph['val'], trn_weights=trn_weights, args=args, param=param)  
        
        elif param['train'] == 'flr':
            train.train_flr(data=data, trn_weights=trn_weights, args=args,param=param)
        
        elif param['train'] == 'xgb':
            train.train_xgb(data=data, trn_weights=trn_weights, args=args, param=param)

        elif param['train'] == 'cnn':
            train.train_cnn(data=data, data_tensor=data_tensor, Y_trn=Y_trn, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)
            
        #elif param['train'] == 'xtx':
        #    train.train_xtx(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, data_kin=data_kin, args=args, param=param)

        elif param['train'] == 'dmlp':
            train.train_dmlp(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)
        
        elif param['train'] == 'lgr':
            train.train_lgr(X_trn=X_trn,  Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)
                
        elif param['train'] == 'dmax':
            train.train_dmax(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)

        elif param['train'] == 'flow':
            train.train_flow(data=data, trn_weights=trn_weights, args=args, param=param)

        elif param['train'] == 'cut':
            None
        elif param['train'] == 'cutset':
            None
        else:
            raise Exception(__name__ + f'.Unknown param["train"] = {param["train"]} for ID = {ID}')

    return


def evaluate_models(data=None, data_tensor=None, data_kin=None, data_graph=None, weights=None, args=None):
    """
    Evaluate ML/AI models.
    """
    print(__name__ + ".evaluate_models: Evaluating models")
    if weights is not None: print(__name__ + " -- per event weighted evaluation ON ")
    print('')
    
    # -----------------------------
    # ** GLOBALS **

    global roc_mstats
    global roc_labels

    mva_mstats = []
    mva_mstats = []
    targetdir  = None

    # -----------------------------
    # Prepare output folders

    outputname = args['rootname']

    targetdir  = f'./figs/{outputname}/{args["config"]}/eval/'
    os.makedirs(targetdir, exist_ok = True)
    
    args["modeldir"] = f'./checkpoint/{outputname}/{args["config"]}/'
    os.makedirs(args["modeldir"], exist_ok = True)

    # --------------------------------------------------------------------
    # Collect data

    X_RAW    = data.tst.x
    X        = copy.deepcopy(data.tst.x)

    y        = data.tst.y
    X_kin    = data_kin.tst.x

    if data_tensor is not None:
        X_2D     = data_tensor['tst']

    if data_graph is not None:
        X_graph  = data_graph['tst']
    
    VARS_kin = data_kin.ids
    # --------------------------------------------------------------------

    print(__name__ + ": Input with {} events and {} dimensions ".format(X.shape[0], X.shape[1]))  

    try:
        ### Tensor variable normalization
        if data_tensor is not None and (args['varnorm_tensor'] == 'zscore'):

            print('\nZ-score normalizing tensor variables ...')
            X_mu_tensor, X_std_tensor = pickle.load(open(args["modeldir"] + '/zscore_tensor.dat', 'rb'))
            X_2D = io.apply_zscore_tensor(X_2D, X_mu_tensor, X_std_tensor)
        
        ### Variable normalization
        if args['varnorm'] == 'zscore':

            print('\nZ-score normalizing variables ...')
            X_mu, X_std = pickle.load(open(args["modeldir"] + '/zscore.dat', 'rb'))
            X = io.apply_zscore(X, X_mu, X_std)

        elif args['varnorm'] == 'madscore':

            print('\nMAD-score normalizing variables ...')
            X_m, X_mad = pickle.load(open(args["modeldir"] + '/madscore.dat', 'rb'))
            X = io.apply_madscore(X, X_m, X_mad)

    except:
        cprint('\n' + __name__ + f' WARNING: {sys.exc_info()[0]} in normalization. Continue without! \n', 'red')
    
    # --------------------------------------------------------------------
    # For pytorch based
    X_ptr    = torch.from_numpy(X).type(torch.FloatTensor)
    if data_tensor is not None:
        X_2D_ptr = torch.from_numpy(X_2D).type(torch.FloatTensor)
    # --------------------------------------------------------------------


    # ====================================================================
    
    def plot_AUC_wrap(func_predict, X, label):
        """ AUC-plot wrapper function.
        """
        for i in range(100):
            try:
                var   = args['plot_param'][f'plot_ROC_binned__{i}']['var']
                edges = args['plot_param'][f'plot_ROC_binned__{i}']['edges']
            except:
                return # No more plots 

            if   len(var) == 1:
                met_1D, label_1D = plots.binned_1D_AUC(func_predict=func_predict, X=X, y=y, weights=weights, X_kin=X_kin, \
                    VARS_kin=VARS_kin, edges=edges, label=label, ids=var[0])

                plots.ROC_plot(met_1D, label_1D, \
                    title = f'<{label}> | training re-weight reference: class ' + str(args['reweight_param']['reference_class']),
                    filename=targetdir + '/' + label + f'__ROC_binned__{i}')

            elif len(var) == 2:

                fig, ax, met = plots.binned_2D_AUC(func_predict=func_predict, X=X, y=y, weights=weights, X_kin=X_kin, \
                    VARS_kin=VARS_kin, edges_A=edges[0], edges_B=edges[1], label=label, ids=var)

                plt.savefig(targetdir + '/' + label + f'__ROC_binned__{i}.pdf', bbox_inches='tight')

                roc_mstats.append(met)
                roc_labels.append(label)

            else:
                print(var)
                raise Exception(__name__ + f'.plot_AUC_wrap: Unknown dimensionality {len(var)}')

    # ====================================================================

    def plot_MVA_wrap(func_predict, X, label, hist_edges):
        """ MVA classifier output density plotter wrapper function.
        """
        fig, ax = plots.density_MVA_output(func_predict=func_predict, X=X, y=y, weights=weights, \
            label=f'<{label}>', hist_edges=hist_edges)
        plt.savefig( targetdir + '/' + label + '_MVA_output.pdf', bbox_inches='tight')

    # ====================================================================
    #
    # **  Loop over active models **
    #

    for i in range(len(args['active_models'])):

        ID = args['active_models'][i]
        param = args[f'{ID}_param']
        print(f'Evaluating <{ID}> | {param} \n')
        
        if   param['predict'] == 'torch_graph':
            func_predict = predict.pred_torch_graph(args=args, param=param)
            plot_AUC_wrap(func_predict=func_predict, X=X_graph, label=param['label'])
            plot_MVA_wrap(func_predict=func_predict, X=X_graph, label=param['label'], hist_edges=np.linspace(0,1,80))

        elif param['predict'] == 'graph_xgb':
            func_predict = predict.pred_graph_xgb(args=args, param=param)
            plot_AUC_wrap(func_predict = func_predict, X = X_graph, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X_graph, label = param['label'], hist_edges=np.linspace(0,1,80))

        elif param['predict'] == 'flr':
            func_predict = predict.pred_flr(args=args, param=param)
            plot_AUC_wrap(func_predict = func_predict, X = X, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X, label = param['label'], hist_edges=80)

        elif param['predict'] == 'xgb':
            func_predict = predict.pred_xgb(args=args, param=param)
            plot_AUC_wrap(func_predict = func_predict, X = X, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X, label = param['label'], hist_edges=np.linspace(0,1,80))

        elif param['predict'] == 'torch_image':
            func_predict = predict.pred_torch(args=args, param=param)

            X_ = {}
            X_tensor['x'] = X_2D_ptr # image tensors
            X_tensor['u'] = X_ptr    # global features
            
            plot_AUC_wrap(func_predict = func_predict, X = X_tensor, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X_tensor, label = param['label'], hist_edges=np.linspace(0,1,80))

        #elif param['predict'] == 'xtx':
        # ...   
        #
        
        elif param['predict'] == 'torch_generic':
            func_predict = predict.pred_torch(args=args, param=param)
            plot_AUC_wrap(func_predict = func_predict, X = X_ptr, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X_ptr, label = param['label'], hist_edges=np.linspace(0,1,80))
        
        elif param['predict'] == 'torch_flow':
            func_predict = predict.pred_flow(args=args, param=param, n_dims=X_ptr.shape[1])
            plot_AUC_wrap(func_predict = func_predict, X = X_ptr, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X_ptr, label = param['label'], hist_edges=80)
        
        elif param['predict'] == 'cut':
            func_predict = predict.pred_cut(args=args, param=param)
            plot_AUC_wrap(func_predict = func_predict, X = X_RAW, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X_RAW, label = param['label'], hist_edges=80)
        
        elif param['predict'] == 'cutset':
            func_predict = predict.pred_cutset(args=args, param=param)
            plot_AUC_wrap(func_predict = func_predict, X = X_RAW, label = param['label'])
            plot_MVA_wrap(func_predict = func_predict, X = X_RAW, label = param['label'], hist_edges=80)

        else:
            raise Exception(__name__ + f'.Unknown param["predict"] = {param["predict"]} for ID = {ID}')

    ### Plot all ROC curves
    plots.ROC_plot(roc_mstats, roc_labels, \
        title = 'training re-weight reference: class ' + str(args['reweight_param']['reference_class']),
        filename = targetdir + 'ROC')

    ### Plot all MVA outputs
    """
    plots.MVA_plot(mva_mstats, mva_labels, \
        title = 'training re-weight reference_class: ' + str(args['reweight_param']['reference_class']),
        filename = targetdir + 'MVA')    
    """

    return


def compute_predictions(func_predict, X):
    """
    Compute predictions
    """
    if type(X) is list:  # Evaluate one by one

        print(__name__ + f'.compute_predictions: one by one evaluation required')
        y_pred = np.zeros(len(X))

        for k in range(len(y_pred)):
            y_pred[k] = func_predict(X[k])
    else:
        y_pred = func_predict(X)

    return y_pred
