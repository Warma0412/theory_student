import numpy as np
import pandas as pd
import pickle
import os
import sys
from torchvision import datasets, transforms
import src
from src.gaussian import GaussianKnockoffs
from src.machine import KnockoffGenerator
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LassoCV
import torch
import torch_two_sample
import torch.nn as nn
from knockpy import knockoff_stats
import copy
from sklearn.model_selection import train_test_split
from collections import defaultdict
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import argparse
from sklearn.neighbors import KernelDensity

import torch
from core_funcs import DataGenerator, eval_fdr_n_power, plot_fdr_n_power_given_df, get_distances_results_df, calculate_vif, filter_df, select
from core_funcs import KnockoffGenerator as KnockoffGeneratorDIY
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn import linear_model
from collections import defaultdict
from tqdm import tqdm
import pickle
import seaborn as sns
import json
from knockpy import knockoff_stats
from IPython.display import display
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.stats import norm, uniform, expon, beta, poisson, binom, lognorm, gamma
warnings.simplefilter(action='ignore', category=FutureWarning)

from evaluation_bridge_sw_trail import bridge

def data_normalizer(data):
    means = data.mean(dim=0, keepdim=True)
    stds = data.std(dim=0, keepdim=True)
    normalized_data = (data - means) / stds
    return normalized_data

def pretrain_reconstruction_compare_plots(x, rec_x):
    fig = plt.figure(layout='constrained', figsize=(16, 20))
    subfigs = fig.subfigures(2, 2, wspace=0.07)
    n, d = x.shape

    nominal_fdrs=[0.1, 0.2, 0.3]
    signal_ns = [20, 60, 80]
    for row in range(2):
        for col in range(2):
            cur_fig = subfigs[row, col]
            cur_ax = cur_fig.add_subplot(111)
            cur_ax.plot(x[row*2+col], label='x')
            cur_ax.plot(rec_x[row*2+col], label='rec_x')
            cur_ax.legend()
            
            
    return fig
            
    
def kernel_density_estimation_plots(X, title):
    # Perform kernel density estimation
    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    kde.fit(X)

    # Generate a grid of points for density estimation
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute the estimated density values
    density_values = np.exp(kde.score_samples(grid_points))
    density_values = density_values.reshape(xx.shape)

    # Plot the original dataset and the estimated density
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=10, color='b', alpha=0.5)
    plt.contourf(xx, yy, density_values, cmap='hot', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Kernel Density Estimation of Two-Moon Dataset {}'.format(title))
    plt.colorbar(label='Density')
    plt.tight_layout()
    return fig

def gen_data_multi_dim(distType="GaussianMixtureAR1", n=2000, d=200,
                       working_dir = './benchmarks/', reset=False, save=False):
    print('creating test data')
    data_path = working_dir + '{}_{}_{}.npy'.format(distType, n, d)
    if save and (not os.path.exists(data_path) or reset):
        # get X
        generator = DataGenerator(n=n, p=d, seed=np.random.randint(10000))
        xTrain, _ = generator.generate_x({'covmethod': distType, 'k': [5]}, data_dir=None)
        xTest, _ = generator.generate_x({'covmethod': distType, 'k': [5]}, data_dir=None)

        to_save_data = {}
        to_save_data['train'] = xTrain
        to_save_data['test'] = xTest
        np.save(data_path, to_save_data)
    else:
        print('skip the line!')

            
    print('done')
    return 


def process_digits(dataset):
    labels = dataset.targets
    indices = (labels == 1) | (labels == 4)
    subset_images = dataset.data[indices]
    subset_labels = dataset.targets[indices]
    subset_labels[subset_labels == 4] = 0

    return subset_images, subset_labels



def load_data_multi_dim(distType="GaussianMixtureAR1", n=2000, d=200,
                       working_dir = './benchmarks/'):
    print('loading test data')
    if distType == 'mnist':
        print('MNIST')
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # Mean and Std deviation of MNIST
        
        trainset = datasets.MNIST('./.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        

        train_data, train_label = process_digits(trainset)
        train_data = train_data.view(-1, 28**2).numpy().astype(np.float32) / 255.0 
        SigmaHat = np.cov(train_data, rowvar=False)
        
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(train_data,0), method="equi", cal_Ds=True)
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / (np.diag(SigmaHat) + 1e-6)
        # Download and load the test data
        testset = datasets.MNIST('./.pytorch/MNIST_data/', download=True, train=False, transform=transform)
        test_data, test_label = process_digits(testset)
        test_data = test_data.view(-1, 28**2).numpy().astype(np.float32) / 255.0

        return (train_data, train_label), (test_data, test_label), corr_g, second_order
    elif distType == 'ibd' or distType == 'ibd_semi':
        data_dict = np.load("./IBD_ST0923dataClean.npy", allow_pickle=True).item()
        data_arr1 = torch.tensor(data_dict["UCvsControl_XX"], dtype=torch.float)
        data_arr2 = torch.tensor(data_dict["CDvsControl_XX"], dtype=torch.float)
        label_arr1 = torch.tensor(data_dict["UCvsControl_YY"], dtype=torch.long)
        label_arr2 = torch.tensor(data_dict["CDvsControl_YY"], dtype=torch.long)
        data_arr = torch.cat([data_arr1, data_arr2], axis=0).numpy()
        label_arr = torch.cat([label_arr1, label_arr2], axis=0)
        label_arr[label_arr==0] = 1
        label_arr[label_arr==2] = 0
        label_arr = label_arr.numpy()
        SigmaHat = np.cov(data_arr, rowvar=False)
        
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data_arr,0), method="equi", cal_Ds=True)
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / (np.diag(SigmaHat) + 1e-6)
        if distType == 'ibd':
            return (data_arr, label_arr), (data_arr, label_arr), corr_g, second_order
        else:
            return data_arr, data_arr, corr_g, second_order
    elif distType == 'rna_semi':
        data_arr = np.load(f'./processed_normalized_subset_rna_{n}_100_no_noise.npy') # 10000 by 100
        SigmaHat = np.cov(data_arr, rowvar=False)
        
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data_arr,0), method="equi", cal_Ds=True)
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / (np.diag(SigmaHat) + 1e-6)
        return data_arr, data_arr, corr_g, second_order
    else:
        data_path = working_dir + '{}_{}_{}.npy'.format(distType, n, d)

        loaded_data = np.load(data_path, allow_pickle=True).item()
        xTrain = loaded_data['train']
        assert xTrain.shape[1] == d and xTrain.shape[0] == n
        xTestFull = loaded_data['test']

        SigmaHat = np.cov(xTrain, rowvar=False)
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(xTrain,0), method="sdp", cal_Ds=distType in ['ar1', 'equiGaussian', 'ark'])
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

        print('done')
        return xTrain, xTestFull, corr_g, second_order



def train_model(xTrain, d, n, distType=None, method=None, second_order=None, corr_g=None, 
                args_dict_sw=None):
    assert method is not None
    if distType in ['mnist', 'ibd']:
        xTrain, xLabel = xTrain
    if 'deepkf' in method:
        assert corr_g is not None
        assert distType is not None
        gamma = 1
        pars={"epochs":200, 
              "epoch_length": 50, 
              "d": d,
              "dim_h": int(6*d),
              "batch_size": min(int(n/4), 512), 
              "lr": 0.01,
              "lr_milestones": [100],
              "GAMMA":gamma, 
              "losstype": 'mmd',
              "epsilon":None,
              "target_corr": corr_g,
              "sigmas":[1.,2.,4.,8.,16.,32.,64.,128.],
              'test_ratio': 0.2
             }
        machine = KnockoffGenerator(pars)
        machine.train(xTrain)
    if 'second' in method:
        assert second_order is not None
        machine = second_order
    if 'gan' in method and 'wgan' not in method:
        sys.path.append('./soft-rank-energy-and-applications/benchmark/')
        from knockoffGAN import knockoffgan
        import tensorflow as tf
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        xTrain, _ = train_test_split(xTrain, test_size=0.2, random_state=42)
        xTr_tilde, machine= knockoffgan(xTrain, xTrain)
    if 'sw' in method:
        
        if distType == 'mnist':
            args_dict = {'n_swapper': 2,
                     'tau': 0.2,
                     'lr_s': 1e-3,
                     'lr_g': 1e-5,
                     'dropout': 0.1,
                     'num_epoch': 500,
                     'batch_size': 16, # 32
                     'destroy_decor_coef': 0.0,
                     'gsw_coeff': 1.,
                     'decor_coeff': 1.,
                     'betaRx': 30.,
                     'input_feature_dim': xTrain.shape[1],
                     'swapper_steps': 3, 
                     'swapper_pen_coeff': 1.,
                     'org_decor_coeff': 1.,
                     'sigma': corr_g,
                     'learn_dist': False,
                     'test_ratio': 0.2,
                        'SWC': False,
                     'linear_hidden_size': 512,
                     'depth': 8} 
        
        else:
            if args_dict_sw is None:
                args_dict = {'n_swapper': 2,
                             'tau': 0.2,
                             'lr_s': 1e-3,
                             'lr_g': 1e-5,
                             'dropout': 0.1,
                             'num_epoch': 200,
                             'batch_size': 32, # 32
                             'destroy_decor_coef': 0.0,
                             'gsw_coeff': 1.,
                             'decor_coeff': 2.,
                             'betaRx': 30.,
                             'input_feature_dim': xTrain.shape[1],
                             'swapper_steps': 3, 
                             'swapper_pen_coeff': 1.,
                             'org_decor_coeff': 4.,
                             'sigma': corr_g,
                             'learn_dist': False,
                             'test_ratio': 0.2,
                             'SWC': True,
                             'linear_hidden_size': 512,
                             'depth': 8}
            else:
                args_dict = args_dict_sw
                args_dict['sigma'] = corr_g
                args_dict['input_feature_dim'] = xTrain.shape[1]
                
                          
        
        xTr_tilde, machine = bridge(xTrain, args_dict, local_rank=0)
        
    
    if 'rank' in method:
        assert corr_g is not None
        pars={"epochs":200, 
          "epoch_length": 20, 
          "d": d,
          "dim_h": min(int(6*d), 1024),
          "batch_size": min(int(n/8), 64),#32 #int(n/4), 
          "lr": 0.01, 
          "lr_milestones": [100],
          "GAMMA":1, 
          "losstype": 'sRMMD', # {sRE, mmd, sRMMD}
          "epsilon":100,
          "target_corr": corr_g,
          "sigmas":[1.,2.,4.,8.,16.,32.,64.,128.],
          'test_ratio': 0.2
         }
        machine = KnockoffGenerator(pars)
        print('start training')
        machine.train(xTrain)
        print('finished training')


    if 'ddlk' in method:
        sys.path.append('./src/')
        import ddlk
        from ddlk import data, utils, mdn, swap
        import pytorch_lightning as pl
        num_gpus = torch.cuda.device_count()
        gpus = [0] if num_gpus > 0 else None
        BATCH_SIZE=64
        xTrain, xTest = train_test_split(xTrain, test_size=0.2, random_state=42)
        dataset = TensorDataset(torch.tensor(xTrain).float())
        dataset_test = TensorDataset(torch.tensor(xTest).float())
        trainloader  = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=8)
        testloader  = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, num_workers=8)
        ((X_mu, ), (X_sigma, )) = utils.get_two_moments(trainloader)
        hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma)
        hparams = argparse.Namespace(
            X_mu=X_mu,
            X_sigma=X_sigma,
            init_type="default",
            hidden_layers=3,
        )
        q_joint = mdn.MDNJoint(hparams)
        ddlk_callbacks = [EarlyStopping(monitor="val_loss")]
        
        
        trainer = pl.Trainer(max_epochs=50, 
                             num_sanity_val_steps=2, 
                             deterministic=True, gpus=gpus,
                             callbacks=ddlk_callbacks)
        
        trainer.fit(q_joint,
                    train_dataloaders=trainloader,
                    val_dataloaders=[testloader])
        
        hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma, 
                                     hidden_layers=3, init_type="residual",
                                     early_stopping=True, lr=1e-3)
        q_knockoff = ddlk.ddlk.DDLK(hparams, q_joint=q_joint)
        
        if hparams.early_stopping:
            ddlk_callbacks = [EarlyStopping(monitor="val_loss")]
        else:
            ddlk_callbacks = []
            
        trainer = pl.Trainer(
            max_epochs=200,
            num_sanity_val_steps=2,
            deterministic=True,
            gradient_clip_val=0.5,
            gpus=gpus,
            callbacks=ddlk_callbacks,
        )
        trainer.fit(q_knockoff,
                train_dataloaders=trainloader,
                val_dataloaders=[testloader])
        xTr, = utils.extract_data(trainloader)
        with torch.no_grad():
            xTr_tilde = q_knockoff.sample(torch.tensor(xTr)).cpu().numpy()
        
        class SamplerNet(nn.Module):
            def __init__(self, net):
                super(SamplerNet, self).__init__()
                self.net = net
                
            def forward(self, inputs):
                with torch.no_grad():
                    xTr_tilde = self.net.sample(torch.tensor(inputs).float()).detach().cpu().numpy()
                return xTr_tilde
        machine = SamplerNet(q_knockoff)
        xTrain = xTr.detach().cpu().numpy()
            
    
    
    if not any([item in method for item in ['sw', 'gan', 'ddlk']]):
        xTr_tilde= machine.generate(xTrain)
    
    return xTrain, xTr_tilde, machine
    
    

def check_swap_property(X, X_knockoff, swap_idx_set, test="FR"):
    feature_dim = X.shape[1]
    assert feature_dim == X_knockoff.shape[1]
    merged_X = torch.cat([X, X_knockoff], dim=1)
    X[:, swap_idx_set], X_knockoff[:, swap_idx_set] = X_knockoff[:, swap_idx_set], X[:, swap_idx_set]
    swaped_merged_X = torch.cat([X, X_knockoff], dim=1)
    return two_sample_tests(merged_X, swaped_merged_X, test)

def tensor_swap(X_all, swap_idx_set):
    feature_dim = X_all.shape[1]//2
    X_all[:, swap_idx_set], X_all[:, swap_idx_set+feature_dim] = X_all[:, swap_idx_set+feature_dim], X_all[:, swap_idx_set]
    return X_all


def two_sample_tests(sampleA, sampleB, test='FR'):
    if test == "FR":
        FSTAT = torch_two_sample.statistics_nondiff.FRStatistic(sampleA.shape[0], sampleB.shape[0])
        stats, statmat = FSTAT.__call__(sampleA, sampleB, ret_matrix=True)
        return FSTAT.pval(statmat)
    if test == "MMD":
        FSTAT = torch_two_sample.statistics_diff.MMDStatistic(sampleA.shape[0], sampleB.shape[0])
        stats, statmat = FSTAT.__call__(sampleA, sampleB, alphas=[1.], ret_matrix=True)
        return FSTAT.pval(statmat)
    if test == 'Energy':
        FSTAT = torch_two_sample.statistics_diff.EnergyStatistic(sampleA.shape[0], sampleB.shape[0])
        stats, statmat = FSTAT.__call__(sampleA, sampleB, ret_matrix=True)
        return FSTAT.pval(statmat)
    if 'KNN' == test[:3]:
        FSTAT = torch_two_sample.statistics_nondiff.KNNStatistic(sampleA.shape[0], sampleB.shape[0], 
                                                              int(test.split('_')[-1]))
        stats, statmat = FSTAT.__call__(sampleA, sampleB, ret_matrix=True)
        return FSTAT.pval(statmat)
    
    
def plot_dist_random_two_entries(xTr, xTr_tilde, sample_corr_test_FR=None):
    aa = copy.deepcopy(xTr)
    bb = copy.deepcopy(xTr_tilde)

    shuffle_idx = np.random.choice(range(aa.shape[1]), int(aa.shape[1]*0.3), replace=False)
    aa[:, shuffle_idx], bb[:, shuffle_idx] = bb[:, shuffle_idx], aa[:, shuffle_idx]
    swaped_merged_X = torch.cat([aa, bb], dim=1)

    plot_idx = [np.random.choice(shuffle_idx, 1, replace=False)[0], 
                np.random.choice(list(set(range(aa.shape[1])).difference(set(shuffle_idx))), 1, replace=False)[0]]
    xmin, xmax = -15, 55
    ymin, ymax = -15, 55

    swaped_merged_X = swaped_merged_X[:, plot_idx]
    # Extract x and y
    x = swaped_merged_X[:, 0]
    y = swaped_merged_X[:, 1]


    # Peform kernel density estimate
    A, B = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([A.ravel(), B.ravel()])

    fig = plt.figure(figsize=(16,16))
    axarr = [fig.gca()]
    # plot data
    kde_data = KernelDensity(bandwidth=6)
    kde_data.fit(swaped_merged_X)
    f = np.exp(np.reshape(kde_data.score_samples(positions.T), A.shape))
    cfset = axarr[0].contourf(A, B, f, cmap='coolwarm')
    cset = axarr[0].contour(A, B, f, colors='k')
    axarr[0].clabel(cset, inline=1, fontsize=10)

    axarr[0].set_title('{}_{}'.format(plot_idx, sample_corr_test_FR), fontsize=20)
    return 

def distribution_check(xTr, xTr_tilde, check_ratio_list=[0.7, 0.8, 0.9, 1.0]):
    print('check distribution')
    check_list = defaultdict(list)
    for replace_ratio in check_ratio_list:
        replace_num = int(replace_ratio * xTr.shape[1])
        for i in range(2):
            tmp_val_stat_p = check_swap_property(copy.deepcopy(xTr), copy.deepcopy(xTr_tilde), np.random.choice(range(xTr.shape[1]), replace_num, replace=False))
            check_list[str(replace_num)].append(tmp_val_stat_p)
    out_comp = {}
    for rep, res in check_list.items():
        out_comp[str(rep)] = np.mean(res)
    print(out_comp)
    return out_comp

def random_single_dim_check(xTr, xTr_tilde):
    rand_show_idx = np.random.choice(range(xTr.shape[1]), 1, replace=False)
    _ = plt.hist(xTr[:, rand_show_idx].numpy(), bins=50)
    _ = plt.hist(xTr_tilde[:, rand_show_idx].numpy(), bins=50)
    rand_show_idx
    
    
def sample_Y(X, signal_n=20, signal_a=10.0, if_complex=False):
    n,p = X.shape
    beta = np.zeros((p,1))
    beta_nonzero = np.random.choice(p, signal_n, replace=False)
    if if_complex:
        beta[beta_nonzero,0] = np.random.randn(signal_n)* signal_a / np.sqrt(n) 
    else:
        beta[beta_nonzero,0] = (2*np.random.choice(2,signal_n)-1) * signal_a / np.sqrt(n)
    y = np.dot(X,beta) + np.random.normal(size=(n,1))
    return y,beta

def calculate_snr(signal, noise):
    signal_power = np.sum(signal**2, axis=1) / signal.shape[1]
    noise_power = np.sum(noise**2, axis=1) / noise.shape[1]
    snr = 10 * np.log10(signal_power / noise_power)
    return np.mean(snr)

def sample_Y_V2(X, signal_n=20, signal_a=10.0, if_complex=False):
    n,p = X.shape
    beta = np.zeros((p,1))
    beta_nonzero = np.random.choice(p, signal_n, replace=False)
    if if_complex:
        beta[beta_nonzero,0] = np.random.randn(signal_n)* signal_a / np.sqrt(n) 
    else:
        beta[beta_nonzero,0] = (2*np.random.choice(2,signal_n)-1) * signal_a / np.sqrt(n)
    signal_part = np.dot(X,beta)
    noise_part = np.random.normal(size=(n,1))
    y = signal_part + noise_part
    return y,beta, calculate_snr(signal_part, noise_part)


def save_model_and_data(machine, data_dict, save_path, method):
    if method in ['deepkf', 'gan', 'ddlk', 'rank']:
        torch.save(machine.net.state_dict(), save_path + '_model.pt')
    if method in ['wgan']:
        model_to_save = machine.module if hasattr(machine, 'module') else machine
        torch.save(model_to_save.state_dict(), save_path + '_model.pt')
    torch.save(data_dict, save_path + '_data.pt')
    return 

def just_save_model(working_dir, file_prefix, machine):
    save_path = working_dir + 'pretrained_{}'.format(file_prefix)
    model_to_save = machine.module if hasattr(machine, 'module') else machine
    torch.save(model_to_save.state_dict(), save_path + '_model.pt')
    return 
    

def just_load_model(working_dir, model_name, device):
    loaded_model = torch.load(working_dir+model_name, map_location=torch.device(device))
    return loaded_model


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    
def fdr_exp(xTestInput, n = 2000, d = 200, distType="GaussianMixtureAR1", 
            working_dir = './benchmarks/',
            complex_y_model=False, method=None, machine=None,
            nominal_fdr=0.1, true_feature_n=20, n_repeats=100,
            lasso_coeff=.01, signal_amplitude_vec=[3, 5, 10, 15, 20, 25, 30], reset=False, appendix=None, 
            save_model=False, xTr_tilde=None, xTr=None, lambda_val_list=[0.85], 
            y_cond_mean='linear', y_coeff_dist='uniform', all_registers=False, fs_model='ridge'):
    assert method is not None
    assert machine is not None
    scaler = StandardScaler()
    print('true feature v.s. all feature: {} / {}'.format(true_feature_n, d))
    # Initialize table of results
    results = pd.DataFrame(columns=['Model','Experiment', 'Method', 'FDP', 'Power', \
                                    'Amplitude', 'Signals', 'Alpha', 'FDR.nominal'])
    if appendix is not None:
        version='{}_{}_{}_{}_{}_{}_{}_{}'.format(method, n, d, 'complex' if complex_y_model else 'simple',
                                        true_feature_n, n_repeats, distType, appendix)
    else:
        version='{}_{}_{}_{}_{}_{}_{}'.format(method, n, d, 'complex' if complex_y_model else 'simple',
                                        true_feature_n, n_repeats, distType)
    print(f'method: {method}')
    print(f'total sample: {n}')
    print(f'feature dim: {d}')
    print(f'y model complexity: {complex_y_model}')
    print(f'n underlying features: {true_feature_n}')
    print(f'experiment repeats: {n_repeats}')
    signal_n = true_feature_n # of nonzero entries
    alpha = lasso_coeff
    file_saved_path = working_dir + 'benchmarks_{}.csv'.format(version)
    xTestLabel = None
    if distType in ['mnist', 'ibd']:
        xTestInput, xTestLabel = xTestInput
    
    if save_model:
        assert xTr is not None and xTr_tilde is not None
        print('save model mode is on, no fdr test will be considered')
        model_n_data_save_prefix = working_dir + '/trained_data/{}'.format(version)
        data_dict = {'train_x_tilde': xTr, 'train_x': xTr_tilde, 'test_x': xTestInput}
        save_model_and_data(machine, data_dict, model_n_data_save_prefix, method)
        return
    if not os.path.exists(file_saved_path) or reset:
        print('writting file')
        
        X_test = torch.tensor(xTestInput).float()
                    
        dataset_test = TensorDataset(X_test)
        data_loader_test = DataLoader(dataset=dataset_test, batch_size=32 if distType != 'mnist' else 16, num_workers=8,
                                     drop_last=False, shuffle=False)

                    
        simple_method_checklist = ["perm", "dist", "X_n_x", "X_n_perm", "mix", "mix_perm", 'mvr', 'sdp']
    
     
        
        Xbase = None
        if method not in simple_method_checklist:
            xk_list = []
            x_hat_list = []
            for xTest in data_loader_test:
                xTest = xTest[0].numpy()
                if not any(item in method for item in ['sw', 'gan', 'ddlk', 'wgan']):
                    xsRMMD= machine.generate(xTest)
                else:
#                     if method == 'gan':
                    if 'gan' in method:
                        xsRMMD= np.concatenate(machine(xTest), axis=0)
                    if 'ddlk' in method:
                        xsRMMD = machine(xTest)
                    if 'wgan' in method:
                        xsRMMD = machine(xTest.unsqueeze(-1).clone(),
                                torch.ones(xTest.shape[0], xTest.shape[1]+1).long().to(xTest.device)).cpu().detach().numpy()
                
                    if 'sw' in method:
                        device = next(machine.parameters()).device
                        xTest = torch.tensor(xTest).to(device)
                        z_test = torch.rand(*xTest.unsqueeze(-1).shape).to(device)
                        xsRMMD = machine(xTest.unsqueeze(-1).clone(),
                                torch.zeros(xTest.shape[0],
                                           xTest.shape[1]+1).long().to(device), z=z_test)
                        if machine.num_classes > 1:
                            xhats = machine(xTest.unsqueeze(-1).clone(),
                                    torch.ones(xTest.shape[0],
                                               xTest.shape[1]+1).long().to(device), z=z_test)
                            x_hat_list.append(xhats.detach().cpu().numpy())
                        xsRMMD = xsRMMD.detach().cpu().numpy()
                        
                xk_list.append(xsRMMD)
                        
            xKTest = np.vstack(xk_list)
            assert xKTest.shape[0] == xTestInput.shape[0]
            Xbase = xKTest
            
            
            check_swap_ratios = [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            dist_df = get_distances_results_df(torch.tensor(xTestInput).float(), torch.tensor(xKTest).float(),
                                               swap_ratios=check_swap_ratios)
            for cur_swap_ratio in check_swap_ratios:
                check_string = dist_df[dist_df['swap_ratio'] == cur_swap_ratio].iloc[0].to_string(header=True, index=True)
                print("pairwise exchangeability at swap ratio: {}: {}".format(cur_swap_ratio, check_string))
                    
        
        assert lambda_val_list is not None
        for LAMBDAVAL in lambda_val_list:
            print('*'*30, f'lambda: {LAMBDAVAL}', '*'*30)
            if method in simple_method_checklist and Xbase is None:
                
                if method == 'perm':
                    N = xTestInput.shape[0]
                    xKTest =  xTestInput.astype(np.float64)[torch.randperm(N).numpy(), ...]
                elif method == 'X_n_perm':
                    N = xTestInput.shape[0]
                    xKTest =  LAMBDAVAL * xTestInput.astype(np.float64) + (1.-LAMBDAVAL)*xTestInput.astype(np.float64)[torch.randperm(N).numpy(), ...]
                elif method == 'X_n_x':
                    generator_ = DataGenerator(n=xTestInput.shape[0], p=xTestInput.shape[1], seed=np.random.randint(10000))
                    Xk2, _ = generator_.generate_x({'covmethod': distType, 'k': [5]}, data_dir=None)
                    xKTest =  LAMBDAVAL * xTestInput.astype(np.float64) + (1.-LAMBDAVAL)*Xk2
                elif method == 'mix':
                    lambda_val = LAMBDAVAL
                    kfgenerator = KnockoffGeneratorDIY(S_method='mvr', method='mx')
                    Xk1 = kfgenerator.generate_knockoff(xTestInput, np.cov(xTestInput.T))

                    generator_ = DataGenerator(n=xTestInput.shape[0], p=xTestInput.shape[1], seed=np.random.randint(10000))
                    Xk2, _ = generator_.generate_x({'covmethod': distType, 'k': [5]}, data_dir=None)

                    xKTest = lambda_val * Xk1 + (1. - lambda_val) * Xk2

                elif method == 'mix_perm':
                    lambda_val = LAMBDAVAL
                    N = xTestInput.shape[0]
                    kfgenerator = KnockoffGeneratorDIY(S_method='mvr', method='mx')
                    Xk1 = kfgenerator.generate_knockoff(xTestInput, np.cov(xTestInput.T))

                    Xk2 =  xTestInput.astype(np.float64)[torch.randperm(N).numpy(), ...]

                    xKTest = lambda_val * Xk1 + (1. - lambda_val) * Xk2

                elif method in ['mvr', 'sdp']:
                    kfgenerator = KnockoffGeneratorDIY(S_method=method, method='mx')
                    xKTest = kfgenerator.generate_knockoff(xTestInput, np.cov(xTestInput.T))
                elif method == 'dist':
                    generator_ = DataGenerator(n=xTestInput.shape[0], p=xTestInput.shape[1], seed=np.random.randint(10000))
                    xKTest, _ = generator_.generate_x({'covmethod': distType, 'k': [5]}, data_dir=None)
                else:
                    assert 1 == 0
            elif method not in simple_method_checklist and Xbase is not None:
                print('#'*30)
                print('in mixture mode')
                print('#'*30)
                if 'mix_perm' in method:
                    N = xTestInput.shape[0]
                    random_state = np.random.RandomState(42)
                    if len(x_hat_list) == 0:
                        xKTest = LAMBDAVAL * Xbase + (1. - LAMBDAVAL) * xTestInput.astype(np.float64)[random_state.permutation(N), ...]
                    else:
                        print('sw dist mode')
                        xKTest = LAMBDAVAL * Xbase + (1. - LAMBDAVAL) * np.vstack(x_hat_list)
            else:

                assert 1 == 0
            
            check_swap_ratios = [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            dist_df = get_distances_results_df(torch.tensor(xTestInput).float(), torch.tensor(xKTest).float(),
                                               swap_ratios=check_swap_ratios)
            for cur_swap_ratio in check_swap_ratios:
                check_string = dist_df[dist_df['swap_ratio'] == cur_swap_ratio].iloc[0].to_string(header=True, index=True)
                print("pairwise exchangeability after permutation at swap ratio: {}: {}".format(cur_swap_ratio, check_string))

            if not all_registers:
                recorder = check_fn(xTestInput, xKTest, method=method,
                                    dataset=distType, N=f'{xTestInput.shape[0]}',
                                   n_repeats=n_repeats, yLabel=xTestLabel,
                                   y_cond_mean=y_cond_mean, y_coeff_dist=y_coeff_dist, fs_model=fs_model)

                if xTestLabel is not None:
                    results = summary_result_noGT(recorder, distType, [method], check_N=xTestInput.shape[0], Lambda=LAMBDAVAL)
                else:
                    results = summary_result(recorder, distType, [method], check_N=xTestInput.shape[0], Lambda=LAMBDAVAL)
            else:
                recorder = check_fn_all_registers(xTestInput, xKTest, method=method,
                                    dataset=distType, N=f'{xTestInput.shape[0]}',
                                   n_repeats=n_repeats, yLabel=xTestLabel,
                                   y_cond_mean=y_cond_mean, y_coeff_dist=y_coeff_dist,
                                                 thresholds=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25], fs_model=fs_model)

                if xTestLabel is not None:
                    results = summary_result_noGT_all_registers(recorder, distType, [method], check_N=xTestInput.shape[0], Lambda=LAMBDAVAL)
                else:
                    results = summary_result_all_registers(recorder, distType, [method], check_N=xTestInput.shape[0], Lambda=LAMBDAVAL)


            print('saving df file')
            insert_string = f"lambda_{LAMBDAVAL}"
            file_saved_path_tmp = file_saved_path.replace('.csv', '_{}.csv'.format(insert_string))
            results.to_csv(file_saved_path_tmp, index=False)
            print('the file saved')
    
            fig = None #plt.figure(figsize=(10, 10))
        else:
            fig = plt.figure(figsize=(10, 10))
            print('skipped! file "{}" exists already. Set reset=True to override'.format(file_saved_path))
    
    return fig



def check_fn(X, Xk, method, dataset, N, n_repeats=600, yLabel=None, 
             y_cond_mean='linear', y_coeff_dist='uniform', fs_model='ridge'):
    recorder = defaultdict(list)

    power_tracker = 0.
    fdp_tracker = 1.
    
    # get generator for Y
    generator = DataGenerator(n=X.shape[0], p=X.shape[1], seed=np.random.randint(10000))
        
    selected_list = []
    for i in tqdm(range(n_repeats), desc='repeats'):


        X = X.astype(np.float64)
        Xk = Xk.astype(np.float64)
        concat_x_test = np.concatenate([X, Xk], axis=1)
        
        if yLabel is not None:
            beta = np.zeros(X.shape[1])
            y = yLabel
            
        else:
            # get y
            y, beta, X, Xk = generator.generate_y(X, Xk, 
                            {'cond_mean': y_cond_mean,
                              'y_dist': ['gaussian'],
                              'sparsity': [0.2],
                              'coeff_dist': [y_coeff_dist]})

        if fs_model == 'ridge':
            RidgeRegressor = knockoff_stats.RidgeStatistic()
        
        if fs_model == 'lasso':
            RidgeRegressor = knockoff_stats.LassoStatistic()
        
        if fs_model == 'ols':
            RidgeRegressor = knockoff_stats.OLSStatistic()
            
        if fs_model == 'rf':
            RidgeRegressor = knockoff_stats.RandomForestStatistic()
        
        if fs_model == 'deeppink':
            RidgeRegressor = knockoff_stats.DeepPinkStatistic()

        RidgeRegressor.fit(X, Xk, y)
        recorder[f'{method}_{dataset}_{N}'].append(RidgeRegressor.Z)
        recorder[f'{method}_{dataset}_{N}_beta'].append(beta)
        
#         if y is None:
        selected, fdp_tmp, power_tmp = select(RidgeRegressor.W, beta, nominal_fdr=0.1, withGT = yLabel is None)
        if yLabel is not None:
            recorder[f'{method}_{dataset}_{N}_selected_score'].append(selected)

        recorder[f'{method}_{dataset}_{N}_power_fdp'].append([pairwise_correlation(X, Xk),
                                                              [power_tmp, fdp_tmp]])
    
#     if y is not None:
#         avg_selected_score = np.mean(np.vstack(selected_list), axis=0)
        
    
    return recorder



def check_fn_all_registers(X, Xk, method, dataset, N, n_repeats=600, yLabel=None, 
             y_cond_mean='linear', y_coeff_dist='uniform',
                           thresholds=[0.01, 0.05, 0.1, 0.15, 0.2], fs_model='ridge'):
    recorder = {}

    power_tracker = 0.
    fdp_tracker = 1.
    
    # get generator for Y
    generator = DataGenerator(n=X.shape[0], p=X.shape[1], seed=np.random.randint(10000))
        
    selected_list = []
    for i in tqdm(range(n_repeats), desc='repeats'):


        X = X.astype(np.float64)
        Xk = Xk.astype(np.float64)
        concat_x_test = np.concatenate([X, Xk], axis=1)
        
        if yLabel is not None:
            beta = np.zeros(X.shape[1])
            y = yLabel
            
        else:
            # get y
            y, beta, X, Xk = generator.generate_y(X, Xk, 
                            {'cond_mean': y_cond_mean,
                              'y_dist': ['gaussian'],
                              'sparsity': [0.2],
                              'coeff_dist': [y_coeff_dist]})

        if fs_model == 'ridge':
            RidgeRegressor = knockoff_stats.RidgeStatistic()
            RidgeRegressor.fit(X, Xk, y)
        
        if fs_model == 'lasso':
            RidgeRegressor = knockoff_stats.LassoStatistic()
            RidgeRegressor.fit(X, Xk, y)
        
        if 'lasso' in fs_model and '_' in fs_model:
            RidgeRegressor = knockoff_stats.LassoStatistic()
            Cs_val = float(fs_model.split('_')[-1])
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp, **{'alphas': [Cs_val]})
        
        if 'ridge' in fs_model and '_' in fs_model:
            RidgeRegressor = knockoff_stats.RidgeStatistic()
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp)
        
        if fs_model == 'ols':
            RidgeRegressor = knockoff_stats.OLSStatistic()
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp)
            
        if fs_model == 'rf':
            RidgeRegressor = knockoff_stats.RandomForestStatistic()
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp)
        
        if fs_model == 'deeppink':
            RidgeRegressor = knockoff_stats.DeepPinkStatistic()
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp, feature_importance='deeppink')
            
        if fs_model == 'deeppink_unweighted':
            RidgeRegressor = knockoff_stats.DeepPinkStatistic()
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp, feature_importance='unweighted')
            
        if fs_model == 'deeppink_swap':
            RidgeRegressor = knockoff_stats.DeepPinkStatistic()
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp, feature_importance='swap')
            
        if fs_model == 'deeppink_swapint':
            RidgeRegressor = knockoff_stats.DeepPinkStatistic()
            n = X.shape[0]
            perm = np.random.permutation(n)
            X_tmp, Xk_tmp, y_tmp = X[perm], Xk[perm], y[perm]
            RidgeRegressor.fit(X_tmp, Xk_tmp, y_tmp, feature_importance='swapint')

        
        if f'{method}_{dataset}_{N}' not in recorder:
            recorder[f'{method}_{dataset}_{N}'] = []
        if f'{method}_{dataset}_{N}_W' not in recorder:
            recorder[f'{method}_{dataset}_{N}_W'] = []
        if f'{method}_{dataset}_{N}_beta' not in recorder:
            recorder[f'{method}_{dataset}_{N}_beta'] = []
        if f'{method}_{dataset}_{N}_selected_score' not in recorder:
            recorder[f'{method}_{dataset}_{N}_selected_score'] = {}
            for threshold in thresholds:
                if str(threshold) not in recorder[f'{method}_{dataset}_{N}_selected_score']:
                    recorder[f'{method}_{dataset}_{N}_selected_score'][str(threshold)] = []
        if f'{method}_{dataset}_{N}_power_fdp' not in recorder:
            recorder[f'{method}_{dataset}_{N}_power_fdp'] = {}
            for threshold in thresholds:
                if str(threshold) not in recorder[f'{method}_{dataset}_{N}_power_fdp']:
                    recorder[f'{method}_{dataset}_{N}_power_fdp'][str(threshold)] = []
            
        
        recorder[f'{method}_{dataset}_{N}'].append(RidgeRegressor.Z)
        recorder[f'{method}_{dataset}_{N}_beta'].append(beta)
        recorder[f'{method}_{dataset}_{N}_W'].append(RidgeRegressor.W)
        
#         if y is None:
        tmp_saver_box = {}
        for threshold in thresholds:
            selected, fdp_tmp, power_tmp = select(RidgeRegressor.W, beta, nominal_fdr=threshold,
                                                  withGT = yLabel is None)
            if yLabel is not None:
                recorder[f'{method}_{dataset}_{N}_selected_score'][str(threshold)].append(selected)
            else:
                recorder[f'{method}_{dataset}_{N}_power_fdp'][str(threshold)].append([pairwise_correlation(X, Xk),
                                                              [power_tmp, fdp_tmp]])
#         if yLabel is not None:
#             recorder[f'{method}_{dataset}_{N}_selected_score'].append(tmp_saver_box)
#         else:
#             recorder[f'{method}_{dataset}_{N}_power_fdp'].append(tmp_saver_box)
    
#     if y is not None:
#         avg_selected_score = np.mean(np.vstack(selected_list), axis=0)
        
    
    return recorder




def pairwise_correlation(array1, array2):
    # Ensure arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape.")

    n, p = array1.shape  # Get the number of rows (n) and columns (p)
    correlations = np.empty(p)  # Array to store correlations

    for i in range(p):
        correlations[i] = np.corrcoef(array1[:, i], array2[:, i])[0, 1]

    return correlations

# # To load it back:
# with open('check_ridge.pkl', 'rb') as f:
#     recorder = pickle.load(f)
def get_corr(vec1, vec2, idx_list):
    all_cor = []

    for ii in idx_list:
        correlation_matrix = np.corrcoef(vec1[:,ii] ,vec2[:, ii])
        sample_correlation = correlation_matrix[0, 1]
        all_cor.append(sample_correlation)
    return np.round(np.max(all_cor), decimals=3), np.round(np.min(all_cor), decimals=3), np.round(np.mean(all_cor), decimals=3)

def get_corrV2(arr, idx_set):

    if isinstance(idx_set, float):
        arr_tmp = arr[:, idx_set]
    else:
        arr_tmp = []
        for a, i in zip(arr, idx_set):
            arr_tmp.append(a[i])
        arr_tmp = np.vstack(arr_tmp)
    return np.round(np.max(arr_tmp), decimals=3), np.round(np.min(arr_tmp), decimals=3), np.round(np.mean(arr_tmp), decimals=3)


def vis_results(recorder, check_method, check_dataset, check_N, individual_dist=False, if_plot=False):
    print("#"*20)
    print(f"{check_method}_{check_dataset}_{check_N}")
    print("#"*20)
    B_shape = np.vstack(recorder[f'{check_method}_{check_dataset}_{check_N}']).shape
    W = np.abs(np.vstack(recorder[f'{check_method}_{check_dataset}_{check_N}'])[:, :B_shape[1]//2]) - np.abs(np.vstack(recorder[f'{check_method}_{check_dataset}_{check_N}'])[:, B_shape[1]//2:])
    W_mean = np.mean(W, axis=0)
    tmp_beta = recorder[f'{check_method}_{check_dataset}_{check_N}_beta']

    if np.asarray(tmp_beta).shape[0] == 1:
        beta = tmp_beta[0]
    else:
        beta = tmp_beta
#     beta = recorder[f'{check_method}_{check_dataset}_{check_N}_beta']

    
    nonzeros_indices = np.where(beta != 0.)[0]

        
    fdps = []
    powers = []
    for j in range(len(W)):
        if not isinstance(beta[0], float):
            selected, fdp, power = select(W[j], beta[j], nominal_fdr=0.1)
        else:
            selected, fdp, power = select(W[j], beta, nominal_fdr=0.1)
        powers.append(power)
        fdps.append(fdp)
    if isinstance(beta[0], float):
        _, fdp_m, power_m = select(W_mean, beta, nominal_fdr=0.1)
        print(f'with mean W: power: {power_m}, fdp: {fdp_m}')
    print(f'power: {np.mean(powers)}, fdp: {np.mean(fdps)}')
    print(f'MIN: power: {np.min(powers)}, fdp: {np.min(fdps)}')
    print(f'MAX: power: {np.max(powers)}, fdp: {np.max(fdps)}')
    if if_plot:
        box_plots = plt.boxplot(W, vert=True, patch_artist=True)

        # Default and highlight colors
        default_color = 'blue'
        highlight_color = 'red'

        # Loop to set colors
        for i, box in enumerate(box_plots['boxes']):
            if i in nonzeros_indices:  # Adding 1 because indices in Python start from 0
                box.set_facecolor(highlight_color)
            else:
                box.set_facecolor(default_color)

        plt.ylabel('Value')
        plt.xlabel('Dimension Index')
        plt.title('Boxplot for Each Dimension with Highlighted Boxes')



        if isinstance(beta[0], float) and individual_dist:
            plot_single_dim_dist(X[:, np.random.choice(nonzeros_indices, 9, replace=False)])

            plot_single_dim_dist(X[:, np.random.choice(np.array(list(set(range(100)) - set(nonzeros_indices))), 9, replace=False)])

        # Display the current plot
        display(plt.gcf())
        plt.close()

    return recorder[f'{check_method}_{check_dataset}_{check_N}_power_fdp'], beta

def plot_correlations(ref, correlation, appendix=''):
        x = ref
        y = correlation
        # Create a DataFrame from the array
        df = pd.DataFrame(data=y, 
                          columns=[f"Column {i+1}" for i in range(y.shape[1])])

        # Create the line plot with shading and red median line using Seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(6, 6))

        for column in df.columns:
            sns.lineplot(data=df[column], x=x, y=df[column], color="gray", lw=1, alpha=0.2)

        sns.lineplot(x=x, y=np.median(y, axis=1), color="red", lw=2)

        plt.title(f"Line Plot with Shaded 90% Interval and Red Median Line ({appendix})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(["90% Interval", "Median"])

        # Display the current plot
        display(plt.gcf())
        plt.close()

        
def W_statistics_cal(W, beta):
    # Ensuring W and beta are numpy arrays
    W = np.asarray(W)
    beta = np.asarray(beta)

    # Validate shapes
    if W.shape != beta.shape:
        raise ValueError("W and beta must have the same shape")

    # Group W based on beta values
    group1 = W[beta != 0]  # Non-zero entries in beta
    group2 = W[beta == 0]  # Zero entries in beta

    # Calculate additional statistics for each group
    stats = {
        'nonnull': {
            'mean': np.mean(group1),
            'median': np.median(group1),
            'std_dev': np.std(group1),
            'variance': np.var(group1),
            'kurtosis': kurtosis(group1),
            'skewness': skew(group1),
            'range': np.ptp(group1),
            'IQR': np.percentile(group1, 75) - np.percentile(group1, 25),
            'count': len(group1),
            'sum': np.sum(group1),
            'MAD': np.mean(np.abs(group1)),
            'min': np.min(group1),
            'max': np.max(group1),
            'percentage_above_threshold': np.mean(np.abs(group1) > 0.1)  # Example with threshold 0.1
        },
        'null': {
            'mean': np.mean(group2),
            'median': np.median(group2),
            'std_dev': np.std(group2),
            'variance': np.var(group2),
            'kurtosis': kurtosis(group2),
            'skewness': skew(group2),
            'range': np.ptp(group2),
            'IQR': np.percentile(group2, 75) - np.percentile(group2, 25),
            'count': len(group2),
            'sum': np.sum(group2),
            'MAD': np.mean(np.abs(group2)),
            'proportion_within_range': np.mean((group2 > -0.1) & (group2 < 0.1))  # Example with range -0.1 to 0.1
        }
    }

    return stats


def summary_result(recorder, check_dataset, method_list, check_N=200, df_=None, Lambda=None):
#     check_dataset = 'ar1'
    if df_ is None:
        df_ = pd.DataFrame(columns=["Dataset", 'Method', 'N', 
                                    'FDP', 'Power', "NullCorr", "NonnullCorr", "Lambda"])

    for check_method in method_list:
#     check_method = 'perm' 
#     choose_idx = 1
#     choose_idx_nonull = np.where(beat_out != 0.)[0][choose_idx]
#     choose_idx_null = np.where(beat_out == 0.)[0][choose_idx]

    
        AA, beta_out = vis_results(recorder, check_method, check_dataset, check_N)
        power_list = []
        fdp_list = []
        correlation_list = []
        for aa in AA:
            correlation_list.append(aa[0])
            power_list.append(aa[1][0])
            fdp_list.append(aa[1][1])
            
#         plot_correlations(np.asarray(power_list), np.vstack(correlation_list), appendix="power")
#         plot_correlations(np.asarray(fdp_list), np.vstack(correlation_list), appendix="fdp")

            nonnull_idx_set_all = np.asarray(beta_out) != 0.#np.where(np.asarray(beta_out) != 0.)[0]
            null_idx_set_all = np.asarray(beta_out) == 0.#np.where(np.asarray(beta_out) == 0.)[0]
        else:
            nonnull_idx_set_all = np.asarray(beta_out) != 0.# np.where(np.asarray(beta_out) != 0.)[0]
            null_idx_set_all = np.asarray(beta_out) == 0. #np.where(np.asarray(beta_out) == 0.)[0]
#         print(np.vstack(correlation_list).shape)
#         print(len(nonnull_idx_set_all))
#         print(len(null_idx_set_all))
#         print(np.vstack(correlation_list).shape)/
#         p

        print(f'nonnull: {get_corrV2(np.vstack(correlation_list), nonnull_idx_set_all)}')
        print(f'null: {get_corrV2(np.vstack(correlation_list), null_idx_set_all)}')
        
        df_ = df_.append({"Dataset": check_dataset, 'Method': check_method, 'N': check_N, 
                                    'FDP': np.mean(fdp_list), 'Power': np.mean(power_list), 
                          "NonnullCorr": get_corrV2(np.vstack(correlation_list), nonnull_idx_set_all),
                          "NullCorr": get_corrV2(np.vstack(correlation_list), null_idx_set_all),
                         "Lambda": Lambda}, ignore_index=True)
        

    return df_




def summary_result_all_registers(recorder, check_dataset,
                                 method_list, check_N=200, df_=None, Lambda=None):
#     check_dataset = 'ar1'
    if df_ is None:
        df_ = pd.DataFrame(columns=["Dataset", 'Method', 'N', 
                                    'FDP', 'Power', "NullCorr", "NonnullCorr", "Lambda",
                                   "Threshold", "STD_FDP", "STD_Power", "Wstat"])

    for check_method in method_list:
    
        AA_dict, beta_out = vis_results(recorder, check_method, check_dataset, check_N)
        for threshold in AA_dict.keys():
            AA = AA_dict[threshold]
            power_list = []
            fdp_list = []
            correlation_list = []
            for aa in AA:
                correlation_list.append(aa[0])
                power_list.append(aa[1][0])
                fdp_list.append(aa[1][1])


                nonnull_idx_set_all = np.asarray(beta_out) != 0.
                null_idx_set_all = np.asarray(beta_out) == 0.
            else:
                nonnull_idx_set_all = np.asarray(beta_out) != 0.
                null_idx_set_all = np.asarray(beta_out) == 0. 
                
            print(f'nonnull: {get_corrV2(np.vstack(correlation_list), nonnull_idx_set_all)}')
            print(f'null: {get_corrV2(np.vstack(correlation_list), null_idx_set_all)}')
            Wstats_data = W_statistics_cal(recorder[f'{check_method}_{check_dataset}_{check_N}_W'], recorder[f'{check_method}_{check_dataset}_{check_N}_beta'])
            df_ = df_.append({"Dataset": check_dataset, 'Method': check_method, 'N': check_N, 
                                        'FDP': np.mean(fdp_list), 'Power': np.mean(power_list), 
                              "NonnullCorr": get_corrV2(np.vstack(correlation_list), nonnull_idx_set_all),
                              "NullCorr": get_corrV2(np.vstack(correlation_list), null_idx_set_all),
                             "Lambda": Lambda, "Threshold": float(threshold),
                             "STD_FDP": np.std(fdp_list), "STD_Power": np.std(power_list),
                             "Wstat": Wstats_data},
                             ignore_index=True)
        
    print("W check")
    print(Wstats_data)
    
    return df_  



def summary_result_noGT(recorder, check_dataset, method_list, check_N=200, df_=None, Lambda=None):
#     check_dataset = 'ar1'
    if df_ is None:
        df_ = pd.DataFrame(columns=["Dataset", 'Method', 'N', 
                                    'Nonnull', "Lambda"])

    for check_method in method_list:
        selected_scores = recorder[f'{check_method}_{check_dataset}_{check_N}_selected_score']
        avg_selected_scores = np.mean(np.vstack(selected_scores), axis=0)
        
        
        df_ = df_.append({"Dataset": check_dataset, 'Method': check_method, 'N': check_N, 
                          "Nonnull": avg_selected_scores, "Lambda": Lambda}, ignore_index=True)
        

    return df_


def summary_result_noGT_all_registers(recorder, check_dataset, method_list, check_N=200, df_=None, Lambda=None):
#     check_dataset = 'ar1'
    if df_ is None:
        df_ = pd.DataFrame(columns=["Dataset", 'Method', 'N', 
                                    'Nonnull', "Lambda", "Threshold"])

    for check_method in method_list:
        
        multi_selected_scores_dict = recorder[f'{check_method}_{check_dataset}_{check_N}_selected_score']
        for threshold in multi_selected_scores_dict.keys():
            selected_scores = multi_selected_scores_dict[threshold]
            avg_selected_scores = np.mean(np.vstack(selected_scores).astype(float), axis=0)
            print('lalala', np.unique(avg_selected_scores), np.vstack(selected_scores).shape)
            df_ = df_.append({"Dataset": check_dataset, 'Method': check_method, 'N': check_N, 
                              "Nonnull": avg_selected_scores, "Lambda": Lambda,
                             "Threshold": float(threshold)}, ignore_index=True)
        

    return df_





