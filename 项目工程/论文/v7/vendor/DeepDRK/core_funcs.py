import numpy as np
import torch
import knockpy.dgp
from scipy import stats
import knockpy
from src import gen_data, oracle, parser, utilities
from knockpy import utilities as knockpy_utilities
from knockpy.knockoff_filter import KnockoffFilter
from knockpy import knockoff_stats as kstats
import time
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
from knockpy import knockoffs
from data_utils import generateSamples
import pandas as pd
from scipy.stats import expon
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import os
import inspect
from src.utils import *
from torch.autograd import grad as torch_grad
import torch.autograd as autograd
import torch.nn.functional as F
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import cvxpy as cvx
from scipy import linalg
import sys
from sklearn.linear_model import LinearRegression

from scipy.stats import norm, uniform, expon, beta, poisson, binom, lognorm, gamma
from pycop import simulation


class CopulaSampler:
    def __init__(self, n, p):
        self.n = n
        self.p = p

    # Sample function
    def sample(self, copula_name, base_distribution, theta=2.0):
        if copula_name not in ['frank', 'joe', 'clayton', 'gumbel']:
            raise ValueError("Invalid copula name")

        Us = simulation.simu_archimedean(copula_name, self.p, self.n, theta=2)
            
        # Convert uniform samples to target distribution
        if base_distribution == "uniform":
            samples = np.column_stack([U for U in Us])
        elif base_distribution == "exponential":
            samples = np.column_stack([expon.ppf(U) for U in Us])
        elif base_distribution == "beta":
            samples = np.column_stack([beta.ppf(U, a=2, b=5) for U in Us])
        elif base_distribution == "gamma":
            samples = np.column_stack([gamma.ppf(U, a=2) for U in Us])
        else:
            raise ValueError("Invalid base distribution")
        
        sample_cov_matrix = np.cov(samples, rowvar=False)
        return samples, sample_cov_matrix

def generate_design_matrix(distribution, n, p):
    if distribution == 'uniform':
        data = uniform.rvs(size=(n, p))
    elif distribution == 'exponential':
        data = expon.rvs(size=(n, p))
    elif distribution == 'beta':
        data = beta.rvs(a=2, b=5, size=(n, p))
    elif distribution == 'poisson':
        data = poisson.rvs(mu=5, size=(n, p))
    elif distribution == 'binomial':
        data = binom.rvs(n=1, p=0.5, size=(n, p))
    elif distribution == 'lognormal':
        data = lognorm.rvs(s=0.5, size=(n, p))
    elif distribution == 'gamma':
        data = gamma.rvs(a=2, size=(n, p))
    else:
        raise ValueError("Invalid distribution name")
    
    sigma = np.cov(data, rowvar=False)  # Calculate sample covariance matrix
    
    return data, sigma



def hinge_loss(a, b):
    """
    Hinge loss to enforce the constraint a >= b.

    Parameters:
        a (torch.Tensor): PyTorch tensor representing value a.
        b (torch.Tensor): PyTorch tensor representing value b.

    Returns:
        loss (torch.Tensor): Hinge loss value.
    """
    return torch.max(torch.tensor(0., dtype=a.dtype, device=a.device), b - a)


def nonzero_zero_indices(beta):
    """
    Return indices of entries that have nonzero values and the indices of entries that are zero.

    Parameters:
        beta (torch.Tensor): PyTorch tensor of shape (p, 1).

    Returns:
        nonzero_indices (torch.Tensor): Indices of entries that have nonzero values.
        zero_indices (torch.Tensor): Indices of entries that are zero.
    """
    nonzero_indices = torch.nonzero(beta).squeeze(1)
    zero_indices = torch.nonzero(beta == 0).squeeze(1)
    return nonzero_indices, zero_indices


def get_ols_penalty_old(X, Xk, sparsity=0.2, gsw_fn=None):
    '''
    n_repeat: the number of samples for the beta distribution
    '''
    n, p = X.shape
    generator = DataGenerator(n=n, p=p, seed=np.random.randint(10000))

    y, beta, X, Xk = generator.generate_y_torch(X, Xk, 
                    {'cond_mean': 'linear',
                      'y_dist': ['gaussian'],
                      'sparsity': [sparsity]})

    y = y.unsqueeze(1)


    beta_mean, beta_covariance = ols_regression(torch.cat([X, Xk], dim=1), y)
    beta_mean_list.append(beta_mean)

    beta_nnull_indices, beta_null_indices = nonzero_zero_indices(beta)
    measure_loss = nn.L1Loss()#torch.nn.MarginRankingLoss(margin=0.1)#nn.L1Loss()
    margin_loss = torch.nn.MarginRankingLoss(margin=0.)
    mean_nnull = margin_loss(beta_mean[beta_nnull_indices],
                             beta_mean[beta_nnull_indices+p], 
                             torch.ones([len(beta_nnull_indices)]).to(X.device))
#     print(mean_nnull)
    if gsw_fn is not None:
        mean_null = gsw_fn.gsw(beta_mean[beta_null_indices], beta_mean[beta_null_indices+p])
    else:
        mean_null = measure_loss(beta_mean[beta_null_indices][..., None], beta_mean[beta_null_indices+p][..., None])
#     print(mean_null)
    mean_loss = mean_null + mean_nnull
#     variance_loss = measure_loss(torch.diag(beta_covariance)[beta_null_indices], 
#                                  torch.diag(beta_covariance)[beta_null_indices+p])
#     print(variance_loss)
    penalty = mean_loss #+ variance_loss
    return penalty



def get_ols_penalty(X, Xk, sparsity=0.1, gsw_fn=None, n_repeat=5, only_null=False):
    '''
    n_repeat: the number of samples for the beta distribution
    '''
    n, p = X.shape
    generator = DataGenerator(n=n, p=p, seed=np.random.randint(10000))
    beta_mean_list = []
    for i in range(n_repeat):
        if i == 0:
            y, beta, X, Xk = generator.generate_y_torch(X, Xk, 
                            {'cond_mean': 'linear',
                              'y_dist': ['gaussian'],
                                  'sparsity': [sparsity]})
        else:
            y, _, X, Xk = generator.generate_y_torch(X, Xk, 
                            {'cond_mean': 'linear',
                              'y_dist': ['gaussian'],
                                  'sparsity': [sparsity]}, beta=beta)
        y = y.unsqueeze(1)
    
    
        beta_mean_tmp, beta_covariance_tmp = ols_regression(torch.cat([X, Xk], dim=1), y)
        beta_mean_tmp = beta_mean_tmp.permute(1, 0)
        beta_mean_list.append(beta_mean_tmp)
    beta_mean = torch.vstack(beta_mean_list)
    beta_nnull_indices, beta_null_indices = nonzero_zero_indices(beta)
    measure_loss = nn.L1Loss()#torch.nn.MarginRankingLoss(margin=0.1)#nn.L1Loss()
    margin_loss = torch.nn.MarginRankingLoss(margin=0.)
    mean_nnull = margin_loss(beta_mean[:, beta_nnull_indices],
                             beta_mean[:, beta_nnull_indices+p], 
                             torch.ones([beta_mean.shape[0], len(beta_nnull_indices)]).to(X.device))

    if gsw_fn is not None:
        mean_null = sliced_wasserstain_distance_individual(beta_mean[:, beta_null_indices], beta_mean[:, beta_null_indices+p],
                                                          gsw_module=gsw_fn)
    else:
        mean_null = measure_loss(beta_mean[:, beta_null_indices], beta_mean[:, beta_null_indices+p])
    if not only_null:
        mean_loss = mean_null + mean_nnull
    else:
        mean_loss = mean_null
    penalty = mean_loss #+ variance_loss
    return penalty



def get_ols_penalty_given_beta(X, Xk, sparsity=0.1, gsw_fn=None, betas=None, only_null=False):
    '''
    n_repeat: the number of samples for the beta distribution
    '''
    assert betas is not None
    n, p = X.shape
    generator = DataGenerator(n=n, p=p, seed=np.random.randint(10000))
    beta_mean_list = []
    for beta in betas: 
        y, _, X, Xk = generator.generate_y_torch(X, Xk, 
                        {'cond_mean': 'linear',
                          'y_dist': ['gaussian'],
                              'sparsity': [sparsity]}, beta=beta)
        y = y.unsqueeze(1)
    
        beta_mean_tmp, beta_covariance_tmp = ols_regression(torch.cat([X, Xk], dim=1), y)
        beta_mean_tmp = beta_mean_tmp.permute(1, 0)
        beta_mean_list.append(beta_mean_tmp)
    beta_mean = torch.vstack(beta_mean_list)
    beta_nnull_indices, beta_null_indices = nonzero_zero_indices(beta)
    measure_loss = nn.L1Loss()#torch.nn.MarginRankingLoss(margin=0.1)#nn.L1Loss()
    margin_loss = torch.nn.MarginRankingLoss(margin=0.)
    mean_nnull = margin_loss(beta_mean[:, beta_nnull_indices],
                             beta_mean[:, beta_nnull_indices+p], 
                             torch.ones([beta_mean.shape[0], len(beta_nnull_indices)]).to(X.device))

    if gsw_fn is not None:
        mean_null = sliced_wasserstain_distance_individual(beta_mean[:, beta_null_indices], beta_mean[:, beta_null_indices+p],
                                                          gsw_module=gsw_fn)
    else:
        mean_null = measure_loss(beta_mean[:, beta_null_indices], beta_mean[:, beta_null_indices+p])
#     print(mean_null, mean_nnull)
    if not only_null:
        mean_loss = mean_null + mean_nnull
    else:
        mean_loss = mean_null
        
    penalty = mean_loss #+ variance_loss
    return penalty


def ols_regression(X, y):
    """
    Perform Ordinary Least Squares (OLS) regression.

    Parameters:
        X (torch.Tensor): Design matrix of shape (N, M), where N is the number of samples and M is the number of features.
        y (torch.Tensor): Output vector of shape (N, 1), where N is the number of samples.

    Returns:
        beta_mean (torch.Tensor): Mean of the coefficients beta of shape (M, 1).
        beta_covariance (torch.Tensor): Covariance matrix of the coefficients beta of shape (M, M).
    """
    # Check the shapes of X and y
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be the same.")

    # Calculate beta_hat (coefficients) using the OLS formula: beta_hat = (X^T * X)^(-1) * X^T * y
    X_transpose = torch.transpose(X, 0, 1)
    XtX_inv = torch.inverse(torch.matmul(X_transpose, X))
    beta_hat = torch.matmul(torch.matmul(XtX_inv, X_transpose), y)

    # Calculate residuals (error vector) epsilon
    epsilon = y - torch.matmul(X, beta_hat)

    # Calculate mean and covariance matrix of beta_hat
    N = X.shape[0]
    M = X.shape[1]
    beta_mean = beta_hat
    sigma_squared = torch.matmul(epsilon.transpose(0, 1), epsilon) / (N - M)
    beta_covariance = XtX_inv * sigma_squared

    return beta_mean, beta_covariance



def solve_sdp(Sigma, tol=1e-3):
    """
    Computes s for sdp-correlated Gaussian knockoffs
    :param Sigma : A covariance matrix (p x p)
    :param mu    : An array of means (p x 1)
    :return: A matrix of knockoff variables (n x p)
    """

    # Convert the covariance matrix to a correlation matrix
    # Check whether Sigma is positive definite
    if(np.min(np.linalg.eigvals(Sigma))<0):
        corrMatrix = cov2cor(Sigma + (1e-8)*np.eye(Sigma.shape[0]))
    else:
        corrMatrix = cov2cor(Sigma)
        
    p,_ = corrMatrix.shape
    s = cvx.Variable(p)
    objective = cvx.Maximize(sum(s))
    constraints = [ 2.0*corrMatrix >> cvx.diag(s) + cvx.diag([tol]*p), 0<=s, s<=1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='CVXOPT')
    
    assert prob.status == cvx.OPTIMAL

    s = np.clip(np.asarray(s.value).flatten(), 0, 1)
    
    # Scale back the results for a covariance matrix
    return np.multiply(s, np.diag(Sigma))

def cal_GaussianKF_Ds(Sigma, method="equi", tol=1e-3):
    # Initialize Gaussian knockoffs by computing either SDP or min(Eigs)

    if method=="equi":
        lambda_min = linalg.eigh(Sigma, eigvals_only=True, eigvals=(0,0))[0]
        s = min(1,2*(lambda_min-tol))
        Ds = np.diag([s]*Sigma.shape[0])
    elif method=="sdp":
        Ds = np.diag(solve_sdp(Sigma,tol=tol))
    else:
        raise ValueError('Invalid Gaussian knockoff type: '+method)
        
    return Ds
            
            
def decorr_loss_cal(X, Xk, Sigma, device, method='equi'):
    Ds = cal_GaussianKF_Ds(Sigma, method=method)
    data_corr = (np.diag(Sigma) - np.diag(Ds)) / np.diag(Sigma)  
    mX  = X  - torch.mean(X,0,keepdim=True)
    mXk = Xk - torch.mean(Xk,0,keepdim=True)
    # Correlation between X and Xk
    eps = 1e-3
    scaleX  = mX.pow(2).mean(0,keepdim=True)
    scaleXk = mXk.pow(2).mean(0,keepdim=True)
    mXs  = mX / (eps+torch.sqrt(scaleX))
    mXks = mXk / (eps+torch.sqrt(scaleXk))
    corr_XXk = (mXs*mXks).mean(0)
    Decorr_loss = (corr_XXk-torch.tensor(data_corr).float().to(device)).pow(2).mean()
    return Decorr_loss


def weights_shifter(weights, FIS_conds):
    return torch.where(FIS_conds > 0., weights, 1.-weights)

        
        
        
def calculate_vif(X, index=None):
    if index is None:
        return [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    else:
        return [variance_inflation_factor(X, index)]


def plot_zjs_n_wjs(A, weights):
    p = len(A)

    # Create figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot A
    colors_a = ['blue' if weight == 0 else 'red' for weight in weights]
    axs[0].scatter(range(p), A, color=colors_a)
    axs[0].set_title('zjs')

    # Plot 1.-A
    colors_1_minus_a = ['green' if weight == 0 else 'red' for weight in weights]
    axs[1].scatter(range(p), 1. - A, color=colors_1_minus_a)
    axs[1].set_title('zjs_tilde')

    # Plot log(A/(1.-A))
    log_ratios = np.log(A / (1. - A))
    colors_log_ratios = ['orange' if weight == 0 else 'red' for weight in weights]
    axs[2].scatter(range(p), log_ratios, color=colors_log_ratios)
    axs[2].set_title('wjs')

    # Set common x-axis label
    plt.xlabel('Index')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Return the figure handler
    return fig

def calculate_loss(y, yhat, method='mse'):
    if method == 'categorical':
        if torch.is_tensor(y) and torch.is_tensor(yhat):
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(yhat, y.squeeze().long())
        else:
            raise ValueError("Both y and yhat should be PyTorch tensors.")
    else:
        if torch.is_tensor(y) and torch.is_tensor(yhat):
            if method == 'mse':
                loss_fn = nn.MSELoss()
            if method == 'l1':
                loss_fn = nn.L1Loss()
            loss = loss_fn(yhat.squeeze(), y.squeeze())
        else:
            raise ValueError("Both y and yhat should be PyTorch tensors.")

    return loss


def bernoulli_weights_normalizer(init_coeff, method='tanh'):
        if method == 'tanh':
            return (0.5*torch.tanh(init_coeff)+0.5)
        if method == 'hardtanh':
            return F.hardtanh(init_coeff, 1e-2, 1.-1e-2)
        if method == 'softsign':
            return (0.5*F.softsign(init_coeff)+0.5)
#         def coeff_NF(self, init_coeff, method='sigmoid'):
        if method == 'sigmoid':
            return F.sigmoid(init_coeff)
        if method == 'minmax':
            maxmax = torch.max(init_coeff)
            minmin = torch.min(init_coeff)
            return ((init_coeff - minmin) / (maxmax-minmin))
        if method == 'max':
            return torch.abs(init_coeff) / torch.abs(init_coeff).max()
        if method == 'softmax':
            return F.softmax(init_coeff) # towards missing values
        if method == 'abs_avg':
            return torch.abs(init_coeff) / torch.abs(init_coeff).sum() # towards more false positives
        if method == 'abs_norm':
            return torch.abs(init_coeff).sum() # towards more false positives
        if method == 'sigmoid_avg':
            return F.sigmoid(init_coeff) / F.sigmoid(init_coeff).sum() # towards more false positives
        if method == 'sigmoid_norm':
            return F.sigmoid(init_coeff).sum() # towards more false positives
        if method == 'square':
            return torch.square(init_coeff) / torch.square(init_coeff).sum()
        
        if method == 'exp':
            return torch.exp(init_coeff) / torch.exp(init_coeff).sum()
        if method == 'mix':
            ratio = .1#0.65 # 0.1
            part_a_softmax = F.softmax(init_coeff)
            part_b_abs_avg = torch.abs(init_coeff) / torch.abs(init_coeff).sum()
            return ratio*part_a_softmax + (1.-ratio)*part_b_abs_avg


def bernoulli_sampling(prob, n, sampling_method='relaxed'):
    """ sampling multinomial given probability """
    if sampling_method == 'continuous':
        dist_tmp = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=prob)
        
    if sampling_method == 'relaxed':
        dist_tmp = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=.1, probs=prob)
    return dist_tmp.rsample((n,))


def improved_gradient_penalty(netD, real_data, fake_data, device):
    lambda_ = .1
    #real_data = real_data.unsqueeze(-1)
    #fake_data = fake_data.unsqueeze(-1)
    #real_data = real_data.permute(1,0,2)
    #fake_data = fake_data.permute(1,0,2)
    batch_size = real_data.size()[0] 
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    
    interpolates = autograd.Variable(interpolates, requires_grad=True).to(device)
    disc_interpolates = netD(interpolates).to(device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, 
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    #real_data = real_data.permute(1,0,2).squeeze()
    #fake_data = fake_data.permute(1,0,2).squeeze()
    return gradient_penalty



def auto_swap_step(xb, xb_tilde, swapper, critic, step=1):
    check = None
    for _ in range(step):
        xb, xb_tilde = swapper(xb, xb_tilde)
        check_tmp = critic(torch.cat([xb, xb_tilde], dim=1)).mean()
        if not check or check_tmp > check: #confirmed
            check = check_tmp
            out_x, out_xtilde = xb, xb_tilde
    return out_x, out_xtilde


def get_distances_results_df(cur_x_pre, cur_x_tilde_pre, swap_ratios=[0., 0.1, 0.3, 0.5 , 0.7, 0.9], cur_model_name=''):
    save_df = pd.DataFrame(columns=['Model','slice_wasserstein_2', 'slice_wasserstein_1', 'mmd_linear', 
                                    'mmd_rbf', 'mmd_poly', 'swap_ratio']
                           )
        
    cur_x = torch.cat([cur_x_pre, cur_x_tilde_pre], axis=1)
    for swap_ratio in swap_ratios:
        if swap_ratio != 0.:
            if swap_ratio != 1.:
                swap_idx_set = np.random.choice(range(cur_x_pre.shape[1]), int(swap_ratio * cur_x_pre.shape[1]), replace=False)
                cur_x_pre[:, swap_idx_set], cur_x_tilde_pre[:, swap_idx_set] = cur_x_tilde_pre[:, swap_idx_set], cur_x_pre[:, swap_idx_set]
            if swap_ratio == 1.:
                cur_x_pre, cur_x_tilde_pre = cur_x_tilde_pre, cur_x_pre
                
        cur_x_tilde = torch.cat([cur_x_pre, cur_x_tilde_pre], axis=1)
        save_df = save_df.append({'Model': cur_model_name,
                        'slice_wasserstein_2': sliced_wasserstein_distance(cur_x, cur_x_tilde, p=2).numpy(),
                        'slice_wasserstein_1': sliced_wasserstein_distance(cur_x, cur_x_tilde, p=1).numpy(),
                                  'mmd_linear': mmd_linear(cur_x.numpy(), cur_x_tilde.numpy()),
                                  'mmd_rbf': mmd_rbf(cur_x.numpy(), cur_x_tilde.numpy()),
                                  'mmd_poly': mmd_poly(cur_x.numpy(), cur_x_tilde.numpy()),
                                  'swap_ratio': swap_ratio,
                                  'sw_dependency': sliced_wasserstain_dependency(cur_x, cur_x_tilde).numpy()}, ignore_index=True)#,
#                                   'sinkhorn': sinkhorn_distance(cur_x.numpy(), cur_x_tilde.numpy(), 1000.)}, ignore_index=True)
        
    return save_df.astype({'Model': 'string','slice_wasserstein_2': 'float32','slice_wasserstein_1': 'float32',
                           'mmd_linear':'float32', 'mmd_rbf': 'float32', 'mmd_poly': 'float32','swap_ratio': 'float32'})
#     sns.histplot(data=save_df, x="slice_wasserstein_1", hue="Model", multiple="dodge", shrink=.8)
    
    
    

def filter_df(df, **kwargs):
    for key, value in kwargs.items():
        df = df[df[key]==value]
    return df


def plot_fdr_n_power_given_df(df, sparsities=[0.2, 0.5, 0.8], kncokoff_methods=['sdp', 'mvr'], nominal_fdr=0.1):

    W_arr = df.W.values[0]

    fig = plt.figure(layout='constrained', figsize=(16, 14))      
    subfigs = fig.subfigures(len(kncokoff_methods), len(sparsities), wspace=0.07)
    for row, kncokoff_method in enumerate(kncokoff_methods):
        for col, sparsity in enumerate(sparsities):
            if len(kncokoff_methods) == 1:
                cur_fig = subfigs[col]
            else:
                cur_fig = subfigs[row, col]
            cur_ax = cur_fig.add_subplot(211)
    #         cur_ax.set_title('nominal FDR: {}, # features: {}'.format(nominal_fdr, signal_n))
            cur_ax.set_title('Xk Method: {}, sparsity: {}/{}'.format(kncokoff_method, 
                                                                       int(sparsity*len(W_arr)),
                                                                       len(W_arr)))
            sns.barplot(x="DataType", y="Power", hue="Method", 
                        data=filter_df(df, FDRnominal=nominal_fdr,
                                       Sparsity=sparsity, KnockoffMethod=kncokoff_method), ax=cur_ax)
            # fig
            #pointplot + methods hue for multi model compari
            cur_ax = cur_fig.add_subplot(212)
            # fig,ax = plt.subplots(figsize=(12,6))
            cur_ax.plot([-1, np.max(30.)+1], 
                    [nominal_fdr, nominal_fdr], linewidth=1, linestyle="--", color="red")
            sns.barplot(x="DataType", y="FDP", hue="Method",
                        data=filter_df(df, FDRnominal=nominal_fdr, 
                                       Sparsity=sparsity, KnockoffMethod=kncokoff_method), ax=cur_ax)
    return fig




def plot_fdr_n_power_given_df_hrt(df, sparsities=[0.2], null_x_methods=['diy'], nominal_fdr=0.1):


    fig = plt.figure(layout='constrained', figsize=(16, 14))      
    subfigs = fig.subfigures(len(null_x_methods), len(sparsities), wspace=0.07)
    for row, null_x_method in enumerate(null_x_methods):
        for col, sparsity in enumerate(sparsities):
            if len(null_x_methods) == 1 and len(sparsities) == 1:
                cur_fig = subfigs
            elif len(null_x_methods) == 1:
                cur_fig = subfigs[col]
            elif len(sparsities) == 1:
                cur_fig = subfigs[row]
            else:
                cur_fig = subfigs[row, col]
            cur_ax = cur_fig.add_subplot(211)
            cur_ax.set_title('Xk Method: {}, sparsity: {}/{}'.format(null_x_method, 
                                                                       int(sparsity*len(df.PVals.values[0])),
                                                                       len(df.PVals.values[0])))
            sns.barplot(x="Method", y="Power", 
                        data=filter_df(df, FDRnominal=nominal_fdr,
                                       Sparsity=sparsity, NullXMethod=null_x_method), ax=cur_ax)
            # fig
            #pointplot + methods hue for multi model compari
            cur_ax = cur_fig.add_subplot(212)
            # fig,ax = plt.subplots(figsize=(12,6))
            cur_ax.plot([-1, np.max(30.)+1], 
                    [nominal_fdr, nominal_fdr], linewidth=1, linestyle="--", color="red")
            sns.barplot(x="Method", y="FDP",
                        data=filter_df(df, FDRnominal=nominal_fdr, 
                                       Sparsity=sparsity, NullXMethod=null_x_method), ax=cur_ax)
    return fig






def eval_fdr_n_power(X, Xk, n_repeats=20):
    n_samples, n_ps = X.shape
    n_samples = [n_samples]
    n_ps = [n_ps]

    sparsities = [0.2, 0.5, 0.8]
    nominal_fdrs = [0.1]
    fs_methods = ['lcd', 'ridge', 'ols', 'randomforest'] # there are more, check knockpy docs
    results = pd.DataFrame(columns=["Z", 'W', 'Method', 'KnockoffMethod', 'Nonzero', 
                                    'FDP', 'Power', 'Selected', "Sparsity", "FDRnominal", "DataType", "N", "P"])

    for _ in tqdm(range(n_repeats)):
        for n_sample in n_samples:
            for n_p in n_ps:
                generator = DataGenerator(n=n_sample, p=n_p, seed=np.random.randint(10000))
                for sparsity in sparsities:
                    # get y
                    y, beta, X, Xk = generator.generate_y(X, Xk, 
                                    {'cond_mean': 'linear',
                                      'y_dist': ['gaussian'],
                                      'sparsity': [sparsity]})
                    for fs_method in fs_methods:
                        # FS
                        selector = FeatureSelector(args={'ksampler': ['gaussian'], 'fstat': [fs_method]})
                        for nominal_fdr in nominal_fdrs:
                            # different nominal fdr
                            selected, power, fdp, W, Z = selector.select_features(X, Xk, y, beta, q=nominal_fdr)
                            results = results.append({"Z": Z, 'W': W, 
                                                      'Method': fs_method,
                                                        'KnockoffMethod': 'wgan',
                                                          'Nonzero': np.sum(beta!=0),
                                                            'FDP': fdp,
                                                              'Power': power, 
                                                              'Selected': selected, 
                                                              "Sparsity": sparsity, 
                                                              "FDRnominal":nominal_fdr,
                                                     "DataType": 'model_dependent', 
                                                      "N": n_sample, "P": n_p}, ignore_index=True)
                        
    return results



# Benjamini-hochberg
def bh(p, fdr):
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries)

def select_hrt(selected, beta):
#     selected = np.where(W >= W_threshold)[0]
    nonzero = np.where(beta!=0.)[0]
    TP = len(np.intersect1d(selected, nonzero))
    FP = len(selected) - TP
    FDP = FP / max(TP+FP,1.0)
    POW = TP / max(len(nonzero),1.0)
    return FDP, POW

def benjamini_hochberg(pvalues, alpha, yekutieli=False):
    """
    Bejamini-Hochberg procedure.
    Extracted from the implementation in sklearn.
    """
    n_features = len(pvalues)
    sv = np.sort(pvalues)
    criteria = float(alpha) / n_features * np.arange(1, n_features + 1)
    if yekutieli:
        c_m = np.log(n_features) + np.euler_gamma + 1 / (2 * n_features)
        criteria /= c_m
    selected = sv[sv <= criteria]
    if selected.size == 0:
        return 0.0, np.zeros_like(pvalues, dtype=bool)
    return selected.max(), pvalues <= selected.max()

def eval_fdr_n_power_hrt(X, method='diy', model=None, n_repeats=20, n_K=200, results=None):
    n_samples, n_ps = X.shape
    n_samples = [n_samples]
    n_ps = [n_ps]
    sparsities = [0.2]#, 0.5, 0.8]
    nominal_fdrs = [0.1]
    if not results:
        results = pd.DataFrame(columns=['PVals', 'Method', 'Nonzero', 'NullXMethod',
                                    'FDP', 'Power', 'Selected', "Sparsity", "FDRnominal", "N", "P"])
        
    
    for _ in tqdm(range(n_repeats), desc="Repeats"):
        for n_sample in n_samples:
            for n_p in n_ps:
                generator = DataGenerator(n=n_sample, p=n_p, seed=np.random.randint(10000))
                for sparsity in sparsities:
                    # get y
                    y, beta, X, _ = generator.generate_y(X, X, 
                                    {'cond_mean': 'linear',
                                      'y_dist': ['gaussian'],
                                      'sparsity': [sparsity]})
                    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4)
                       
                    model_for_hrt = LinearRegression().fit(X_train, Y_train)
                    X_test = torch.tensor(X_test).float()
                    Y_test = torch.tensor(Y_test).float()
                    dataset_test = TensorDataset(X_test, Y_test)
                    data_loader_test = DataLoader(dataset=dataset_test, batch_size=32, num_workers=8,
                                                 drop_last=False)
                    tstat_fn = lambda X_eval: ((Y_test - model_for_hrt.predict(X_eval))**2).mean()
                    ref_tstat = tstat_fn(X_test)
                    total_p_vales = []
                    for cur_idx in tqdm(range(n_p), desc="Dimension", leave=False):
                        # get conditional nulls for the x_i
                        tmp_tstat = 0.
                        for _ in tqdm(range(n_K), desc="K cnt", leave=False):
                            # model get conditional samples
                            null_xb = []
                            for xb_test, _ in data_loader_test:
#                                 print(xb_test.shape)
                                xb_test = xb_test.to(model.device)
                                z_test = torch.rand(*xb_test.unsqueeze(-1).shape).to(model.device)

                                seq_label_test = torch.zeros(xb_test.shape[0], X.shape[1]+1).long().to(xb_test.device)
                                seq_label_test[:, cur_idx+1] = 1
                                seq_label_test = seq_label_test.long()
                                null_xb_test = model(xb_test.unsqueeze(-1).clone(),
                                        seq_label_test, z=z_test)
                                null_xb.append(null_xb_test.detach().cpu().numpy())
                            

                            X_eval = np.vstack(null_xb)
                            tmp_tstat += int(tstat_fn(X_eval) <= ref_tstat)
                         
                        total_p_vales.append((1.+tmp_tstat) / (1.+float(n_K)))
                            
                    for nominal_fdr in nominal_fdrs:
    
                        selected, _, _, _ = multipletests(total_p_vales, alpha=nominal_fdr, method='fdr_bh',
                                                         maxiter=1, is_sorted=False, returnsorted=False)
                        threshold, selected = benjamini_hochberg(total_p_vales, nominal_fdr)
                        print('check two bh methods', sum(np.asarray(selected).astype(int) == np.asarray(selected).astype(int)), len(total_p_vales))
                        fdp, power = select_hrt(selected, beta)
                        results = results.append({'PVals': total_p_vales, 
                                                   'NullXMethod': 'diy',
                                                      'Method': 'OLS',
                                                          'Nonzero': np.sum(beta!=0),
                                                            'FDP': fdp,
                                                              'Power': power, 
                                                              'Selected': selected, 
                                                              "Sparsity": sparsity, 
                                                              "FDRnominal":nominal_fdr,
                                                      "N": n_sample, "P": n_p}, ignore_index=True)
    return results
                        





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


def plot_conditional_dist_comparison(samples1, label1='Generated', label2='GT', conditional_kwargs=None):
    assert conditional_kwargs is not None
    mean = conditional_kwargs['mean']
    covariance = conditional_kwargs['covariance']
    condition_indices = conditional_kwargs['condition_indices']
    condition_values = conditional_kwargs['condition_values']

    samples2 = conditional_multivariate_gaussian_sample(len(samples1), mean, covariance, condition_indices, condition_values)
    
    fig, ax = plt.subplots()

    # Plot histograms
    ax.hist(samples1, bins=20, density=True, alpha=0.5, color='blue', label=label1)
    ax.hist(samples2.squeeze(), bins=20, density=True, alpha=0.5, color='orange', label=label2)

    # Add labels and legend
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distributions Comparison')
    ax.legend()
    samples1 = samples1[..., None]
    fig1 = plot_conditional_kde_comparison(samples1, samples2)
    return fig, fig1

def plot_conditional_kde_comparison(samples1, samples2):
    return kernel_density_estimation_plots(np.concatenate([samples1, samples2], axis=1), '{}'.format('kde'))


def plot_swapped_merged_X(xb_test, xb_tilde_test, shuffle_idx=None, shuffle_ratio=0.3, plot_idx=None):
    if shuffle_idx is None:
        shuffle_idx = np.random.choice(range(xb_test.shape[1]), int(xb_test.shape[1]*shuffle_ratio), replace=False)
        
    xb_test[:, shuffle_idx], xb_tilde_test[:, shuffle_idx] = xb_tilde_test[:, shuffle_idx], xb_test[:, shuffle_idx]
    swaped_merged_X = torch.cat([xb_test, xb_tilde_test], dim=1)
    if plot_idx is None:
        plot_idx = [
            np.random.choice(shuffle_idx, 1, replace=False)[0], 
            np.random.choice(list(set(range(xb_test.shape[1])).difference(set(shuffle_idx))), 1, replace=False)[0]
        ]
    
    swaped_merged_X = swaped_merged_X[:, plot_idx].numpy()
    
    fig = kernel_density_estimation_plots(swaped_merged_X, '{}'.format(plot_idx))
    
    return fig, shuffle_idx, plot_idx


def set_model_parameter_gradient(model, if_gradient):
    for p in model.parameters():
        p.requires_grad = if_gradient
    return 
    

def load_model(model, model_weights_path):
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    return model


def save_model(working_dir, file_prefix, machine):
    save_path = working_dir + '/{}.pt'.format(file_prefix)
    model_to_save = machine.module if hasattr(machine, 'module') else machine
    torch.save(model_to_save.state_dict(), save_path)
    return

# def load_model(working_dir, file_prefix, machine):
#     load_path = os.path.join(working_dir, '/{}.pt'.format(file_prefix))
#     model_to_load = machine.module if hasattr(machine, 'module') else machine
#     model_to_load.load_state_dict(torch.load(load_path))
#     print(f"Model loaded from: {load_path}")
#     return machine

def get_init_arguments(model):
    init_args = inspect.signature(model.__init__).parameters
    init_values = {k: v for k, v in model.__dict__.items() if k in init_args}
    return init_values


def conditional_samples(mu, sigma, x_given, indices_given):
    # Split the mean and covariance matrix into the necessary submatrices
    mu_given = mu[indices_given]
    mu_to_predict = mu[~indices_given]
    sigma_given_given = sigma[indices_given][:, indices_given]
    sigma_to_predict_given = sigma[~indices_given][:, indices_given]
    sigma_to_predict_to_predict = sigma[~indices_given][:, ~indices_given]

    # Compute the conditional mean and covariance
    sigma_given_given_inv = torch.inverse(sigma_given_given)
    mu_conditional = mu_to_predict + torch.matmul(
        torch.matmul(sigma_to_predict_given, sigma_given_given_inv),
        (x_given - mu_given)
    )
    sigma_conditional = sigma_to_predict_to_predict - torch.matmul(
        torch.matmul(sigma_to_predict_given, sigma_given_given_inv),
        sigma_to_predict_given.t()
    )

    # Sample from the conditional distribution
    dist_conditional = torch.distributions.MultivariateNormal(mu_conditional, sigma_conditional)
    samples = dist_conditional.sample()
    
    return samples



def data_normalizer(data, means=None, stds=None):
    means = data.mean(axis=0, keepdims=True) if not means else means
    stds = data.std(axis=0, keepdims=True) if not stds else stds
    normalized_data = (data - means) / stds
    return normalized_data, means, stds


def data_normalizer_torch(data, means=None, stds=None, reverse=False):
    means = data.mean(dim=0, keepdim=True) if means is None else means
    stds = data.std(dim=0, keepdim=True) if stds is None else stds
    if not reverse:
        normalized_data = (data - means) / stds
    else:
        normalized_data = data * stds + means
    return normalized_data, means, stds


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


def check_and_create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")


def save_dict_to_pt(dictionary, file_path):
    # Save the dictionary to a Torch .pt file
    torch.save(dictionary, file_path)
    print("Dictionary saved to:", file_path)

def load_dict_from_pt(file_path):
    # Load the dictionary from the Torch .pt file
    dictionary = torch.load(file_path)
    print("Dictionary loaded from:", file_path)
    return dictionary


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def fit_exponential_n_pdf(data):
    # Fit exponential distribution to data
    fit_params = expon.fit(data)

    # Extract estimated scale parameter
    estimated_scale = fit_params[1]

    # Evaluate PDF at specific points
    pdf_at_points = expon.pdf(data, scale=estimated_scale)
    normalized_pdf_values = pdf_at_points / pdf_at_points.max() 
    return normalized_pdf_values


def sample_and_shuffle_data_bernoulli(W, p_values):
    # Bernoulli parameters
    # p_values = normalized_pdf_values  # Example parameter values

    # Number of samples
    num_samples = 1

    # Draw samples from Bernoulli distribution
    samples = np.random.binomial(1, p_values, size=(num_samples, len(p_values)))


    new_W = W * np.sign((samples.squeeze() - 0.5) * -1.)
    return new_W



# assuming gen_data is some external module with gen_ark_X function
# import gen_data 
def kfilter(W, offset=1.0, q=0.1):
    """
    Adaptive significance threshold with the knockoff filter
    :param W: vector of knockoff statistics
    :param offset: equal to one for strict false discovery rate control
    :param q: nominal false discovery rate
    :return a threshold value for which the estimated FDP is less or equal q
    """
    t = np.insert(np.abs(W[W!=0]),0,0)
    t = np.sort(t)
    ratio = np.zeros(len(t));
    for i in range(len(t)):
        ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        
    index = np.where(ratio <= q)[0]
    if len(index)==0:
        thresh = float('inf')
    else:
        thresh = t[index[0]]
       
    return thresh



def select(W, beta, nominal_fdr=0.1, offset = 1.0, withGT=True):
    W_threshold = kfilter(W, offset = offset, q=nominal_fdr)
    if withGT:
        selected = np.where(W >= W_threshold)[0]
    else:
        selected = (W >= W_threshold).astype(float)
    if withGT:
        nonzero = np.where(beta!=0.)[0]
        TP = len(np.intersect1d(selected, nonzero))
        FP = len(selected) - TP
        FDP = FP / max(TP+FP,1.0)
        POW = TP / max(len(nonzero),1.0)
    else:
        FDP = 0.
        POW = 0.
    return selected, FDP, POW


def equi_Gaussian_sampling(n, p):
    # Mean vector: zero for all dimensions
    mean = np.zeros(p)
    
    # Covariance matrix with diagonal ones and off-diagonals 0.3
    covariance = np.full((p, p), 0.3)  # fill the matrix with 0.3
    np.fill_diagonal(covariance, 1.)  # set the diagonal to 1
    
    # Generate samples
    samples = np.random.multivariate_normal(mean, covariance, n)
    
    return samples, covariance

def conditional_multivariate_gaussian_sample(n, mean, covariance, condition_indices, condition_values):
    # Extract the relevant components
    unconditioned_indices = np.delete(np.arange(mean.shape[0]), condition_indices)
    A = covariance[np.ix_(unconditioned_indices, unconditioned_indices)]
    B = covariance[np.ix_(unconditioned_indices, condition_indices)]
    C = covariance[np.ix_(condition_indices, unconditioned_indices)]
    D = covariance[np.ix_(condition_indices, condition_indices)]
    
    # Calculate conditional mean and covariance
    conditional_mean = mean[unconditioned_indices] + np.dot(np.dot(B, np.linalg.inv(D)), condition_values - mean[condition_indices])
    conditional_covariance = A - np.dot(np.dot(B, np.linalg.inv(D)), C)
    
    # Sample from the conditional Gaussian distribution
    sampled_values = np.random.multivariate_normal(conditional_mean, conditional_covariance, size=n)
    
    return sampled_values

def sin5(x):
    return np.sin(5.0 * x)


def cos5(x):
    return np.cos(5.0 * x)


def generate_mu_ddlk(x, beta, option='ddlk2'):
    # Initialize y with zeros for all samples
    n, p = x.shape
    assert len(beta) == p, "The dimension of beta should match the number of columns in x"
    
    y = np.zeros(n)  # This will assume epsilon is 0, if epsilon has a value, replace zeros with that value
    
    # Get indices where beta is non-zero
    non_zero_beta_indices = np.where(beta != 0)[0]
    
    # Assert that there are exactly 20 non-zero entries in beta
    assert len(non_zero_beta_indices) == 20 or len(non_zero_beta_indices) == 16, "beta should have exactly 20 non-zero entries"
    
    m = len(non_zero_beta_indices)  # As given, corresponding to the 20 non-zero entries in beta
    m_over_4 = m // 4
    
    for i in range(m_over_4):
        # Use the non-zero indices from beta to select columns from x
        base_idx = i * 4
        x_4k_3 = x[:, non_zero_beta_indices[base_idx]]
        x_4k_2 = x[:, non_zero_beta_indices[base_idx + 1]]
        x_4k_1 = x[:, non_zero_beta_indices[base_idx + 2]]
        x_4k = x[:, non_zero_beta_indices[base_idx + 3]]
        
        # Sample the phi parameters from normal distributions
        phi_1, phi_2 = np.random.normal(1, 1, 2)
        phi_3, phi_4, phi_5, phi_6 = np.random.normal(2, 1, 4)
        
        # Compute y for this set of columns
        if option[-1] == '2':
            y_k = (phi_1 * x_4k_3 + phi_3 * x_4k_2 + phi_4 * x_4k_3 * x_4k_2 +
                   phi_5 * np.tanh(phi_2 * x_4k_1 + phi_6 * x_4k))
        elif option[-1] == '3':
            y_k = (phi_1 * x_4k_3 + phi_3 * x_4k_2 + phi_4 * x_4k_3 * x_4k_2)
        elif option[-1] == '4':
            y_k = phi_1 * x_4k_3 + phi_3 * x_4k_2 + phi_5 * np.tanh(phi_2 * x_4k_1 + phi_6 * x_4k)
        else:
            assert 1 == 0
        y += y_k
        
    return y


class DataGenerator:
    def __init__(self, n, p, seed=None):
        self.n = n
        self.p = p
        self.seed = seed
        self.data = None
        self.sigma = None
        
    def generate_conditional_Gaussian(self, n, mean, cov, condition_indices, condition_values):
        return conditional_multivariate_gaussian_sample(n, mean, cov, condition_indices, condition_values)


    def generate_x(self, args, data_dir=None):
        if data_dir is not None:
            self.load_torch_data(data_dir)
            return self.data, self.sigma
        np.random.seed(self.seed)
        dgprocess = knockpy.dgp.DGP()
        max_corr = args.get('max_corr', [0.99])[0]
        # print(max_corr)
        sample_kwargs = {}
        covmethod = args.get('covmethod', ['ar1'])
        # print(covmethod)
        if covmethod in ['GaussianAR1', 'GaussianMixtureAR1', 'twomoon'] or ('GaussianMixtureAR1' in covmethod and '_' in covmethod):
            dataSampler = generateSamples(covmethod.split('_')[0] if '_' in covmethod else covmethod, self.p, float(covmethod.split('_')[-1]) if '_' in covmethod else 0.6)
            self.data = dataSampler.sample(self.n)
            self.sigma = knockpy_utilities.estimate_covariance(self.data, tol=1e-2)[0]
        elif covmethod in ['ver', 'ar1', 'blockequi']:
            if covmethod in ['ver', 'ar1']:
                sample_kwargs['max_corr'] = max_corr
            if covmethod == 'ar1':
                sample_kwargs['a'] = args.get("a", [5])[0]
                sample_kwargs['b'] = args.get("b", [1])[0]
            if covmethod == 'blockequi':
                sample_kwargs['rho'] = args.get("rho", [0.5])[0]
                sample_kwargs['gamma'] = args.get("gamma", [0])[0]
            if covmethod == 'ver':
                sample_kwargs['delta'] = args.get("delta", [0.2])[0]
            dgprocess.sample_data(n=self.n, p=self.p, method=covmethod, **sample_kwargs)
            self.data = dgprocess.X
            self.sigma = dgprocess.Sigma
        elif covmethod == 'ark':
            self.data, self.sigma = gen_data.gen_ark_X(n=self.n, p=self.p, k=args.get("k", [2])[0], max_corr=max_corr)
        elif covmethod == 'orthogonal':
            self.data = stats.ortho_group.rvs(self.n)
            self.data = self.data[:, 0:self.p]
            self.sigma = np.eye(self.p)
        elif covmethod == 'equiGaussian':
            self.data, self.sigma = equi_Gaussian_sampling(self.n, self.p)
        elif covmethod.split('_')[0] == 'copula':
            _, copula_name, dist_name = covmethod.split('_')
            copula_sampler = CopulaSampler(n=self.n, p=self.p)
            self.data, self.sigma = copula_sampler.sample(copula_name, dist_name, theta=8.0)
        else:
            print(covmethod)
            try:
                self.data, self.sigma = generate_design_matrix(covmethod, self.n, self.p)
            except:
                raise NotImplementedError
        return self.data, self.sigma
    # write a data visualization function here that visualizes self.data by selection randomly 2 columns of self.data and plot them
    # use matplotlib scatter function to plot the data
    def visualize_data(self, data_, plot_indices=None):
        if plot_indices is None:
            plot_indices = np.random.choice(data_.shape[1], 2, replace=False)
        data =data_[:, plot_indices]  

        # Create a figure and axes.
        fig, ax = plt.subplots()

        # Generate the 2D kernel density plot. You can customize the color map (cmap) as needed.
        sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap="Reds", fill=True, ax=ax, thresh=0, levels=100)

        # Overlay with a scatter plot. You can customize the color and size of the points as needed.
        ax.scatter(data[:, 0], data[:, 1], color='blue', s=5)

        # Display the plot.
        plt.show()

        # also plot the covariance matrix and label it
        sigma = knockpy_utilities.estimate_covariance(data_, tol=1e-2)[0]
        plt.imshow(sigma)
        plt.colorbar()


    def load_torch_data(self, data_dir):
        self.data = torch.load(f'{data_dir}/X.pt')
        self.sigma = knockpy_utilities.estimate_covariance(self.data, tol=1e-2)[0]


    def generate_y(self, X, Xk, args, beta=None):
        # Sample beta
        if beta is None:
            beta = gen_data.sample_beta(
                p=self.p,
                sparsity=args.get("sparsity", [0.2])[0],
                coeff_dist=args.get("coeff_dist", ["uniform"])[0],
                coeff_size=args.get("coeff_size", [1])[0],
                corr_signals=args.get("corr_signals", [False])[0],
                N = X.shape[0]
            )
        # Sample y
        cond_mean = args.get("cond_mean", ["linear"])
        if cond_mean == 'linear':
            fX, fXk = X, Xk
        elif cond_mean == 'cos':
            fX, fXk = np.cos(X), np.cos(Xk)
        elif cond_mean == 'sin':
            fX, fXk = np.sin(X), np.sin(Xk)
        elif cond_mean == 'sin_cos_alter':
#             signal_n = X.shape[0]
            no_zero_idx = beta != 0.
            fX = X
            fXk = Xk
            fX[:, no_zero_idx] = 5. * (np.sin(X[:, no_zero_idx]) + np.cos(X[:, no_zero_idx]))
            fXk[:, no_zero_idx] = 5. * (np.sin(Xk[:, no_zero_idx]) + np.cos(Xk[:, no_zero_idx]))
        elif cond_mean == 'sin_cos5_alter':
#             signal_n = X.shape[0]
            no_zero_idx = beta != 0.
            fX = X
            fXk = Xk
            fX[:, no_zero_idx] = 5. * (sin5(X[:, no_zero_idx]) + cos5(X[:, no_zero_idx]))
            fXk[:, no_zero_idx] = 5. * (sin5(Xk[:, no_zero_idx]) + cos5(Xk[:, no_zero_idx]))
        elif cond_mean == 'ddlk1':
            no_zero_idx = np.where(beta != 0.)[0]
            np.random.shuffle(no_zero_idx)

            # Calculate the size of each subset
            subset_size = len(no_zero_idx) // 3
            
            # Create three subsets
            subset1 = no_zero_idx[:subset_size]
            subset2 = no_zero_idx[subset_size:2*subset_size]
            subset3 = no_zero_idx[2*subset_size:]

            fX = X
            fXk = Xk
            # first set linear
            fX[:, subset1] = X[:, subset1]
            fXk[:, subset1] = Xk[:, subset1]
            # second set second order
            fX[:, subset2] = X[:, subset2] 
            fXk[:, subset2] = Xk[:, subset2]
            
            # third set nonlinear
            fX[:, subset3] = X[:, subset3]
            fXk[:, subset3] = Xk[:, subset3]
            
            
            
        elif cond_mean == 'cosh':
            fX, fXk = np.cosh(X), np.cosh(Xk)
        elif cond_mean == 'sinh':
            fX, fXk = np.sinh(X), np.sinh(Xk)
        elif cond_mean == 'quadratic':
            fX, fXk = X**2, Xk**2
        elif cond_mean == 'cubic':
            fX, fXk = X**3, Xk**3
        elif cond_mean == 'trunclin':
            fX = (X > 0).astype(float)
            fXk = (Xk > 0).astype(float)
        elif cond_mean in ['ddlk2', 'ddlk3', 'ddlk4']:
            fX, fXk = X, Xk
        else:
            raise ValueError(f"unrecognized cond_mean={cond_mean}")
        if cond_mean not in ['ddlk2', 'ddlk3', 'ddlk4']:
            mu = np.dot(fX, beta)
        else:
            mu = generate_mu_ddlk(fX, beta, option=cond_mean)
            
        y = gen_data.sample_y(mu=mu, y_dist=args.get("y_dist", ['gaussian'])[0])
        return y, beta, fX, fXk
    

    def generate_y_torch(self, X, Xk, args, beta=None):
        # Sample beta (assuming gen_data.sample_beta returns a PyTorch tensor)
        if beta is None:
            beta = gen_data.sample_beta_torch(
                p=self.p,
                sparsity=args.get("sparsity", [0.2])[0],
                coeff_dist=args.get("coeff_dist", ["uniform"])[0],
                coeff_size=args.get("coeff_size", [1])[0],
                corr_signals=args.get("corr_signals", [False])[0],
                device=X.device
            )
        else:
            beta = beta.to(X.device)

        # Sample y
        cond_mean = args.get("cond_mean", ["linear"])
        if cond_mean == 'linear':
            fX, fXk = X, Xk
        elif cond_mean == 'cos':
            fX, fXk = torch.cos(X), torch.cos(Xk)
        elif cond_mean == 'sin':
            fX, fXk = torch.sin(X), torch.sin(Xk)
        elif cond_mean == 'cosh':
            fX, fXk = torch.cosh(X), torch.cosh(Xk)
        elif cond_mean == 'sinh':
            fX, fXk = torch.sinh(X), torch.sinh(Xk)
        elif cond_mean == 'quadratic':
            fX, fXk = X**2, Xk**2
        elif cond_mean == 'cubic':
            fX, fXk = X**3, Xk**3
        elif cond_mean == 'trunclin':
            fX = (X > 0).float()
            fXk = (Xk > 0).float()
        else:
            raise ValueError(f"unrecognized cond_mean={cond_mean}")

        mu = torch.matmul(fX, beta)
        y = gen_data.sample_y_torch(mu=mu, y_dist=args.get("y_dist", ['gaussian'])[0], device=X.device)
        return y, beta, fX, fXk




class KnockoffGenerator:
    def __init__(self, S_method, method='mx', seed=42):
        self.S_method = S_method
        self.method = method
        self.seed = seed

    def generate_knockoff(self, X, Sigma):
        # Setup sampler based on self.mx flag
        if self.method == 'mx':
            ksampler = knockoffs.GaussianSampler(X=X, Sigma=Sigma, method=self.S_method)
            Xk = ksampler.sample_knockoffs()
        elif self.method == 'fx':
            ksampler = knockoffs.FXSampler(X=X, method=self.S_method)
            Xk = ksampler.sample_knockoffs()
        else:
            raise ValueError(f"Invalid method {self.method}")

        self.S = ksampler.fetch_S()
        self.Xk = Xk   
        return self.Xk
    
    def get_S(self):
        return self.S
    
    def visualize_data(self):
        plot_indices = np.random.choice(self.Xk.shape[1], 2, replace=False)
        data = self.Xk[:, plot_indices]  

        # Create a figure and axes.
        fig, ax = plt.subplots()

        # Generate the 2D kernel density plot. You can customize the color map (cmap) as needed.
        sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap="Reds", fill=True, ax=ax, thresh=0, levels=100)

        # Overlay with a scatter plot. You can customize the color and size of the points as needed.
        ax.scatter(data[:, 0], data[:, 1], color='blue', s=5)

        # Display the plot.
        plt.show()
        # return Xk


class FeatureSelector:
    def __init__(self, args) -> None:
        self.ksampler = args.get("ksampler", ["gaussian"])[0]
        self.fstat = args.get("fstat", ["lasso"])[0]
        self.args = args
        
    def select_features(self, X, Xk, y, beta, q=0.05):
        if self.fstat not in ['data_split', 'new']:
            # initialize KF
            kf = KnockoffFilter(ksampler=self.ksampler, fstat=self.fstat)
            fstat_kwargs = dict()
            if self.fstat in ['mlr', 'bcd']:# or (self.fstat == 'oracle' and mx):
                fstat_kwargs['n_iter'] = self.args.get("n_iter", [2000])[0]
                fstat_kwargs['chains'] = self.args.get("chains", [5])[0]

            kf.forward(
                X=X, Xk=Xk, y=y, fstat_kwargs=fstat_kwargs, fdr=q,
            )
        

            W = kf.W
            Z = kf.Z

            selected, fdp, power = select(W, beta, nominal_fdr=q)
        else:
            if self.fstat == 'data_split':
                kf1 = KnockoffFilter(ksampler=self.ksampler, fstat='lcd')
                kf2 = KnockoffFilter(ksampler=self.ksampler, fstat='lcd')
                fstat_kwargs = dict()
                half_size = X.shape[0]//2
                p = X.shape[1]
                kf1.forward(X=X[:half_size, ...], Xk=np.zeros_like(X[:half_size, ...]), y=y[:half_size, ...], fstat_kwargs=fstat_kwargs, fdr=q)
                kf2.forward(X=X[half_size:, ...], Xk=np.zeros_like(X[half_size:, ...]), y=y[half_size:, ...], fstat_kwargs=fstat_kwargs, fdr=q)
                Z1 = kf1.Z[:p]
                Z2 = kf2.Z[:p]
                Z = np.concatenate([Z1, Z2], axis=0)
                W = np.sign(Z1*Z2)*(np.abs(Z1)+np.abs(Z2))
            selected, fdp, power = select(W, beta, nominal_fdr=q)

        return selected, power, fdp, W, Z






