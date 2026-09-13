import torch

import os
import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import itertools
import copy
import argparse
import random
import numpy as np
import os
import pandas as pd
import torch_optimizer as optimAdd
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import scipy.stats as st
import torch_two_sample
import math
import logging as loggingPython
import scipy.stats as st
from torch.utils.data.distributed import DistributedSampler

from KFT import KFMAETransformer, GumbelSwapper
from gsw import GSW, sliced_wasserstain_dependency, sliced_wasserstain_dependency_multi, sliced_wasserstain_dependency_n_perm, sliced_wasserstain_dependency_individual, sliced_wasserstain_dependency_1vsOther, SWC
from torchvision import transforms
from core_funcs import DataGenerator, FeatureSelector, KnockoffGenerator, set_random_seeds, save_dict_to_pt, load_dict_from_pt, check_and_create_directory, pretrain_reconstruction_compare_plots, data_normalizer, data_normalizer_torch, get_init_arguments, save_model, load_model, set_model_parameter_gradient, eval_fdr_n_power, get_distances_results_df, auto_swap_step, improved_gradient_penalty, plot_swapped_merged_X, plot_fdr_n_power_given_df, calculate_vif, decorr_loss_cal, get_ols_penalty, get_ols_penalty_given_beta
from sklearn.model_selection import train_test_split
import inspect
import warnings


# Disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Penalize correlations between variables and knockoffs

def org_decor(X, Xk, Sigma):
    # Center X and Xk
    mX  = X  - torch.mean(X,0,keepdim=True)
    mXk = Xk - torch.mean(Xk,0,keepdim=True)
    # Correlation between X and Xk
    eps = 1e-3
    scaleX  = mX.pow(2).mean(0,keepdim=True)
    scaleXk = mXk.pow(2).mean(0,keepdim=True)
    mXs  = mX / (eps+torch.sqrt(scaleX))
    mXks = mXk / (eps+torch.sqrt(scaleXk))
    corr_XXk = (mXs*mXks).mean(0)
    loss_corr = (corr_XXk-Sigma).pow(2).mean()
    return loss_corr


def bridge(X_train, args_dict, local_rank=0):
    best_val_loss = float('inf')
    patience = 6
    early_stop = False
    
    args = argparse.Namespace(**args_dict)
    X_train, X_test = train_test_split(X_train, test_size=args.test_ratio, random_state=42)
    
    ###################
    # init
    ###################
    
    device = torch.device("cuda:{}".format(local_rank))
    
    
    # init swappers and optimizers
    swapper_all_parameters = []
    swapper_list = []
    swapper_opt_list = []
    for _ in range(args.n_swapper):   
        tmp_swapper = GumbelSwapper(args.input_feature_dim, args.tau).to(device)
        swapper_list.append(tmp_swapper)
        swapper_opt_list.append(torch.optim.Adam(tmp_swapper.parameters(), lr=args.lr_s))
        swapper_all_parameters += list(tmp_swapper.parameters())

    optimizerS = torch.optim.Adam(swapper_all_parameters, lr=args.lr_s)
    swapper_opt_list = []
    for _ in range(args.n_swapper):
        swapper_opt_list.append(optimizerS)


    # init netG and optimizers

    netG = KFMAETransformer(input_length=args.input_feature_dim, num_input_features=1, linear_hidden_size=args.linear_hidden_size,
                     num_classes=1 if not args.learn_dist else 2, depth=args.depth, heads=8, mlp_dim=1024,
                     dim_head = 64, dropout_p = args.dropout, emd_dropout_p = args.dropout).to(device)

    optimizerG = torch.optim.AdamW(netG.parameters(), lr=args.lr_g)
    
    
    # get gsw functions
    gsw_fn = GSW(device, swapper_list, swapper_opt_list, nofprojections=20)
    
    
    #get dataloader
    X_train = torch.tensor(X_train).float()
    dataset = TensorDataset(X_train)
    dataset_test = TensorDataset(torch.tensor(X_test).float())
    
    
    #####################
    # training part
    #####################
    init_epoch = 0
    global_step = 0
    one = torch.ones([])
    one = one.to(device)
    mone = one * -1
    cos_sim = torch.nn.CosineSimilarity(dim=1)

    for epoch in tqdm(range(init_epoch, args.num_epoch+1), desc='epoch'):
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=8)
        for iteration, (xb) in enumerate(data_loader):
            xb = xb[0].to(device)


            # enable random
            z = torch.rand(*xb.unsqueeze(-1).shape).to(xb.device)
            #####################
            # 1. generator train
            #####################
            errG = 0.
            xb_hat = None
            recon_loss = 0.
            if global_step % 1 == 0:
                set_model_parameter_gradient(netG, True)
                for cur_swapper in swapper_list:
                    set_model_parameter_gradient(cur_swapper, False)


                optimizerG.zero_grad()
                
                xb_tilde = netG(xb.unsqueeze(-1).clone(),
                                torch.zeros(xb.shape[0], args.input_feature_dim+1).long().to(xb.device),
                                z=z)

                if args.learn_dist:
                    xb_hat = netG(xb.unsqueeze(-1).clone(),
                                torch.ones(xb.shape[0], args.input_feature_dim+1).long().to(xb.device),
                                z=z)
                
                    recon_loss = gsw_fn.gsw(xb, xb_hat)
                ub_list = []
                ub_tilde_list = []



                gsw_loss_list = gsw_fn.gsw_n_swapper(xb, xb_tilde, opt_swapper=False) # set opt_swapper to False casue we optimizer the swappers in here outside the function
#                 print(gsw_loss_list)
                GSW_loss = torch.mean(torch.stack(gsw_loss_list)) + recon_loss
                


                
                if args.SWC:
                    destroy_decor_loss = args.destroy_decor_coef * 0.5 * (SWC(xb, xb_tilde, max_version=False, gsw_module=gsw_fn, mode='destroy', correlation=args.SWC) + SWC(xb_tilde, xb, max_version=False, gsw_module=gsw_fn, mode='destroy', correlation=args.SWC))
                    Decorr_loss = 0.5*(SWC(xb, xb_tilde, max_version=False, gsw_module=gsw_fn, mode='individual', correlation=args.SWC) + SWC(xb_tilde, xb, max_version=False, gsw_module=gsw_fn, mode='individual', correlation=args.SWC)) + destroy_decor_loss + args.org_decor_coeff * org_decor(xb, xb_tilde, torch.tensor(args.sigma).to(xb.device)) 
                else:
                    Decorr_loss = args.org_decor_coeff * org_decor(xb, xb_tilde, torch.tensor(args.sigma).to(xb.device))
                

                # IRM loss
                if len(swapper_list) > 1:
                    Rx_IRM_loss = torch.var(torch.stack(gsw_loss_list))
                else:
                    Rx_IRM_loss = 0.
                
                

                errG = args.gsw_coeff*GSW_loss + args.decor_coeff* Decorr_loss + args.betaRx * Rx_IRM_loss
                errG.backward(one)
                optimizerG.step()


                #####################
                # 2. swapper train
                #####################
                
                
                errS = 0.
                if_opt_swappers = epoch >= 0 and global_step % args.swapper_steps == 0

                recon_loss = 0.
                ensemble_penalty = 0.
                swapper_loss = 0.
                if if_opt_swappers:
                    set_model_parameter_gradient(netG, False)
                    for cur_swapper in swapper_list:
                        set_model_parameter_gradient(cur_swapper, True)
                    
                    
                    optimizerS.zero_grad()
                    xb_tilde = netG(xb.unsqueeze(-1).clone(),
                                torch.zeros(xb.shape[0], args.input_feature_dim+1).long().to(xb.device),
                                z=z)
                    ub_list = []
                    ub_tilde_list = []

                    
                    if args.learn_dist:
                        xb_hat = netG(xb.unsqueeze(-1).clone(),
                                    torch.ones(xb.shape[0], args.input_feature_dim+1).long().to(xb.device),
                                    z=z)
                        recon_loss = gsw_fn.gsw(xb, xb_hat)



                    gsw_loss_list = gsw_fn.gsw_n_swapper(xb, xb_tilde, opt_swapper=False)

                    GSW_loss = torch.mean(torch.stack(gsw_loss_list)) + recon_loss
                    

                    #push swappers to be far away
                    ensemble_penalty = 0.
                    if len(swapper_list) > 1:
                        for sw_a, sw_b in itertools.combinations(list(range(len(swapper_list))), 2):
                            ensemble_penalty += torch.logsumexp(cos_sim(swapper_list[sw_a].pi_net[:, :],
                                                                        swapper_list[sw_b].pi_net[:, :]), 0).mean()
                    else:
                        ensemble_penalty = 0.
                    swapper_loss = args.swapper_pen_coeff * ensemble_penalty
                    # IRM loss
                    if len(swapper_list) > 1:
                        Rx_IRM_loss = torch.var(torch.stack(gsw_loss_list))
                    else:
                        Rx_IRM_loss = 0.


                    errS = swapper_loss - args.gsw_coeff*GSW_loss - args.betaRx * Rx_IRM_loss #- args.rev_pen_coeff/GSW_loss 
                    errS.backward()
                    optimizerS.step()
                    



            errD = 0.
            global_step += 1
            
            if (epoch % 10 == 0) and local_rank == 0 and iteration==0:
                print('epoch {} iteration{}, G Loss: {}, D Loss: {}, S Loss: {}, GSW: {}, Decor: {}, swapper: {}, IRM: {}, recon loss: {}'.format(epoch, global_step, 
                                                                                                                         
                                                                                        errG.item() if errG != 0. else errG, 
                                                                                        errD.item() if errD != 0. else errD,
                                                                                        errS.item() if errS != 0. else errS,
                                                                                                                         GSW_loss.item() if GSW_loss != 0. else GSW_loss, Decorr_loss.item() if Decorr_loss != 0. else Decorr_loss, swapper_loss.item() if swapper_loss != 0. else swapper_loss, Rx_IRM_loss.item() if Rx_IRM_loss != 0. else Rx_IRM_loss, recon_loss))
                
                
                print('multiplier check', args.gsw_coeff, args.betaRx, args.destroy_decor_coef, args.decor_coeff, args.org_decor_coeff)
                
                data_loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, num_workers=8,
                                     drop_last=False, shuffle=False)
                
                netG.eval()
                with torch.no_grad():
                    x_hat_list = []
                    recon_test = []
                    gsw_test = []
                    decor_test = []
                    IRM_test = []
                    for xTest in data_loader_test:
                        xTest = xTest[0]

                        xTest = torch.tensor(xTest).to(device)
                        z_test = torch.rand(*xTest.unsqueeze(-1).shape).to(device)
                        xsRMMD = netG(xTest.unsqueeze(-1).clone(),
                                torch.zeros(xTest.shape[0],
                                           xTest.shape[1]+1).long().to(device), z=z_test)
                        recon_loss = 0.
                        if args.learn_dist:
                            xhats = netG(xTest.unsqueeze(-1).clone(),
                                    torch.ones(xTest.shape[0],
                                               xTest.shape[1]+1).long().to(device), z=z_test)
                            
                            recon_loss = gsw_fn.gsw(xTest, xhats)


                        if xsRMMD.dim() == 1:
                            xsRMMD = xsRMMD[None, ...]
                                    
                        gsw_loss_list = gsw_fn.gsw_n_swapper(xTest, xsRMMD, opt_swapper=False) # set opt_swapper to False casue we optimizer the swappers in here outside the function
                    
                        if args.SWC:
                            destroy_decor_loss_test = args.destroy_decor_coef * 0.5 * (SWC(xTest, xsRMMD, max_version=False, gsw_module=gsw_fn, mode='destroy', correlation=args.SWC) + SWC(xsRMMD, xTest, max_version=False, gsw_module=gsw_fn, mode='destroy', correlation=args.SWC))
                            Decorr_loss_test = 0.5*(SWC(xTest, xsRMMD, max_version=False, gsw_module=gsw_fn, mode='individual', correlation=args.SWC) + SWC(xsRMMD, xTest, max_version=False, gsw_module=gsw_fn, mode='individual', correlation=args.SWC)) + destroy_decor_loss_test + args.org_decor_coeff * org_decor(xTest, xsRMMD, torch.tensor(args.sigma).to(xTest.device)) # add switched option see if there's differencel
                        else:
                            Decorr_loss_test = args.org_decor_coeff * org_decor(xTest, xsRMMD, torch.tensor(args.sigma).to(xTest.device))
                        gsw_test.append(torch.mean(torch.stack(gsw_loss_list)).item())
                        decor_test.append(Decorr_loss_test.item())
                        recon_test.append(recon_loss.item() if not isinstance(recon_loss, float) else recon_loss)
                        if len(swapper_list) > 1:
                            IRM_test_val = args.betaRx * torch.var(torch.stack(gsw_loss_list)).item()
                        else:
                            IRM_test_val = 0.
                        
                        IRM_test.append(IRM_test_val)
                    avg_val_loss =  np.mean(gsw_test) + np.mean(recon_test) + np.mean(decor_test) + np.mean(IRM_test)
                    
                print(f'testing all: {avg_val_loss}, gsw_test: {np.mean(gsw_test)}, recon: {np.mean(recon_test)}, decor: {np.mean(decor_test)}, IRM: {np.mean(IRM_test)}')
                netG.train()
                # Check if early stopping conditions are met
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_out = netG
                    counter = 0
                else:
                    counter += 1
                    print(f"EarlyStopping counter: {counter} out of {patience}")
                    if counter >= patience:
                        early_stop = True
                        break
        if early_stop:
            print('Early Stopped')
            break
    
    print('training finished')
    if not early_stop:
        model_out = netG
    return X_train, model_out.module if hasattr(model_out, 'module') else model_out