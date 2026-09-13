import os
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from ddlk import utils
# import sys
# sys.path.append('/home/hongyu2/project/small_model_test/TransKG/benchmarks/')
# from data_generation import *
import torch

def generate_num_polynomial_terms(degree):
    terms = []
    for i in range(degree + 1):
        for j in range(i + 1):
            term = (i - j, j)  # Coefficients for the term
            terms.append(term)
    return len(terms)


# def sample_two_moon_high_dim(n_sample=100, n_dim=10):
#     X = make_moons(n_samples=n_sample, noise=0.1)[0]
#     for cur_degree in range(1, n_dim+1):
#         if generate_num_polynomial_terms(cur_degree) > n_dim+1:
#             set_degree = cur_degree
#             break
    

#     poly = PolynomialFeatures(degree=set_degree)
#     X_projected = poly.fit_transform(X)
#     return X_projected[:, 1:n_dim+1].astype(np.float32)



def sample_two_moon_high_dim(n_sample=100, n_dim=10):
    two_moon_data = make_moons(n_samples=n_sample)[0]
    random_projection = np.random.normal(size=[2, n_dim])
    random_projection_std = random_projection/ np.linalg.norm(random_projection, axis=-1)[..., None]
    return (two_moon_data @ random_projection).astype(np.float32)


def sample_multivariate_normal(mu, Sigma, n=1):
    return (np.linalg.cholesky(Sigma) @ np.random.randn(mu.shape[0], n) +
            mu.reshape(-1, 1)).T


def covariance_AR1(p, rho):
    """
    Construct the covariance matrix of a Gaussian AR(1) process
    """
    assert len(
        rho) > 0, "The list of coupling parameters must have non-zero length"
    assert 0 <= max(
        rho) <= 1, "The coupling parameters must be between 0 and 1"
    assert 0 <= min(
        rho) <= 1, "The coupling parameters must be between 0 and 1"

    # Construct the covariance matrix
    Sigma = np.zeros(shape=(p, p))
    for i in range(p):
        for j in range(i, p):
            Sigma[i][j] = reduce(mul, [rho[l] for l in range(i, j)], 1)
    Sigma = np.triu(Sigma) + np.triu(Sigma).T - np.diag(np.diag(Sigma))
    return Sigma


class TwoMoon:
    def __init__(self, p):
        self.p = p
    def sample(self, n=1):
        return sample_two_moon_high_dim(n, self.p)
        


class GaussianAR1:
    """
    Gaussian AR(1) model
    """

    def __init__(self, p, rho, mu=None):
        """
        Constructor
        :param p      : Number of variables
        :param rho    : A coupling parameter
        :return:
        """
        self.p = p
        self.rho = rho
        self.Sigma = covariance_AR1(self.p, [self.rho] * (self.p - 1))
        if mu is None:
            self.mu = np.zeros((self.p, ))
        else:
            self.mu = np.ones((self.p, )) * mu

    def sample(self, n=1, **args):
        """
        Sample the observations from their marginal distribution
        :param n: The number of observations to be sampled (default 1)
        :return: numpy matrix (n x p)
        """
        return sample_multivariate_normal(self.mu, self.Sigma, n)
        # return np.random.multivariate_normal(self.mu, self.Sigma, n)


class GaussianMixtureAR1:
    """
    Gaussian mixture of AR(1) model
    """

    def __init__(self, p, rho_list, mu_list=None, proportions=None):
        # Dimensions
        self.p = p
        # Number of components
        self.K = len(rho_list)
        # Proportions for each Gaussian
        if (proportions is None):
            self.proportions = [1.0 / self.K] * self.K
        else:
            self.proportions = proportions

        if mu_list is None:
            mu_list = [0 for _ in range(self.K)]

        # Initialize Gaussian distributions
        self.normals = []
        # self.Sigma = np.zeros((self.p, self.p))
        for k in range(self.K):
            rho = rho_list[k]
            self.normals.append(GaussianAR1(self.p, rho, mu=mu_list[k]))
            # self.Sigma += self.normals[k].Sigma / self.K

    def sample(self, n=1, **args):
        """
        Sample the observations from their marginal distribution
        :param n: The number of observations to be sampled (default 1)
        :return: numpy matrix (n x p)
        """
        # Sample vector of mixture IDs
        Z = np.random.choice(self.K, n, replace=True, p=self.proportions)
        # Sample multivariate Gaussians
        X = np.zeros((n, self.p))
        for k in range(self.K):
            k_idx = np.where(Z == k)[0]
            n_idx = len(k_idx)
            X[k_idx, :] = self.normals[k].sample(n_idx)
        return X


def signal_Y(X, signal_n=20, signal_a=10.0):
    """
    From: https://github.com/msesia/DeepKnockoffs
    """
    n, p = X.shape

    beta = np.zeros((p, 1))
    beta_nonzero = np.random.choice(p, signal_n, replace=False)
    beta[beta_nonzero,
         0] = (2 * np.random.choice(2, signal_n) - 1) * signal_a / np.sqrt(n)

    y = np.dot(X, beta) + np.random.normal(size=(n, 1))
    return X, y.flatten(), beta


def create_dataloaders(X,
                       Y,
                       beta,
                       batch_size=64,
                       train_size=0.7,
                       test_size=0.15):
    X = X.astype('float32')
    Y = Y.astype('float32')
    if beta is None:
        pass
    else:
        beta = beta.astype('float32')
#  if not None else beta

    xTr, xVal, yTr, yVal = train_test_split(X, Y, test_size=(1 - train_size))
    xTe, xVal, yTe, yVal = train_test_split(xVal,
                                            yVal,
                                            test_size=(2 * test_size))

    trainloader = utils.create_jointloader(xTr,
                                           yTr,
                                           beta,
                                           batch_size=batch_size,
                                           shuffle=True)
    valloader = utils.create_jointloader(xVal,
                                         yVal,
                                         beta,
                                         batch_size=1000,
                                         shuffle=False)
    testloader = utils.create_jointloader(xTe,
                                          yTe,
                                          beta,
                                          batch_size=1000,
                                          shuffle=False)

    return trainloader, valloader, testloader


def get_data_real(args, data, label):
    assert args.rep is not None, 'rep must be an integer'
    input_feature_dim = args.d
    BATCH_SIZE = args.batch_size
    dropout_rate=0.
    if_perm_input = False
    syn_dataset_name = 'syn2' 
    shuffle_check_idx_list = []
    perm_index = torch.randperm(input_feature_dim)
    cur2org = {}
    org2cur = {}
    for i in range(input_feature_dim):
        cur2org[i] = perm_index[i].item()
        org2cur[perm_index[i].item()] = i
#     tmp_x = generate_x(args.n, input_feature_dim)
    # tmp_x_knockoffs = gen_knockoffs(tmp_x)
#     data_arr = torch.tensor(data, dtype=torch.float)/
#     label_arr = torch.tensor(label, dtype=torch.long)
#     label_arr = label_arr[:, 0]

    trainloader, valloader, testloader = create_dataloaders(
        data,
        label,
        None,
        batch_size=args.batch_size,
        train_size=0.7,
        test_size=0.15)
    return trainloader, valloader, testloader
    
def get_data(args):
    """
    Used in experiments.

    Returns trainloader, valloader, testloader for a particular dataset
    """

    assert args.rep is not None, 'rep must be an integer'
    if args.dataset == 'diy':
        input_feature_dim = args.d
        BATCH_SIZE = args.batch_size
        dropout_rate=0.
        if_perm_input = False
        syn_dataset_name = args.syn_dataset_name
        shuffle_check_idx_list = []
        perm_index = torch.randperm(input_feature_dim)
        cur2org = {}
        org2cur = {}
        for i in range(input_feature_dim):
            cur2org[i] = perm_index[i].item()
            org2cur[perm_index[i].item()] = i
        tmp_x = generate_x(args.n, input_feature_dim)
        # tmp_x_knockoffs = gen_knockoffs(tmp_x)
        data_arr = torch.tensor(tmp_x, dtype=torch.float)
        tmp_y = generate_y(tmp_x, syn_dataset_name)
        label_arr = torch.tensor(tmp_y, dtype=torch.long)
        label_arr = label_arr[:, 0]
        
        trainloader, valloader, testloader = create_dataloaders(
            tmp_x,
            tmp_y,
            None,
            batch_size=args.batch_size,
            train_size=0.7,
            test_size=0.15)
        
    elif args.dataset == 'gaussian_autoregressive':

        data_sampler = GaussianAR1(p=args.d, rho=args.rho)

        X = data_sampler.sample(n=args.n)
        _, Y, beta = signal_Y(X, signal_n=args.n_rel, signal_a=args.signal_a)

        trainloader, valloader, testloader = create_dataloaders(
            X,
            Y,
            beta,
            batch_size=args.batch_size,
            train_size=0.7,
            test_size=0.15)

    elif args.dataset == 'gaussian_autoregressive_mixture':

        # number of dimensions
        d = args.d
        # number of mixture components
        k = args.k

        # mixture parameters
        prop = (2 + (0.5 + np.arange(k) - k / 2)**2)**0.9
        prop = prop / prop.sum()
        rho_list = [0.6**(i + 0.9) for i in range(k)]

        data_sampler = GaussianMixtureAR1(p=d,
                                          rho_list=rho_list,
                                          mu_list=[20 * i for i in range(k)],
                                          proportions=prop)

        X = data_sampler.sample(n=args.n)
        _, Y, beta = signal_Y(X, signal_n=args.n_rel, signal_a=args.signal_a)

        trainloader, valloader, testloader = create_dataloaders(
            X,
            Y,
            beta,
            batch_size=args.batch_size,
            train_size=0.7,
            test_size=0.15)
    else:
        raise NotImplementedError(
            f'Dataset {args.dataset} is not implemented...')

    return trainloader, valloader, testloader























































