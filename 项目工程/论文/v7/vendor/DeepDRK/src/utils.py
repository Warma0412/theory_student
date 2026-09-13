# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:33:02 2022

@author: shoaib
"""
import numpy as np
import math
import sys

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from scipy.spatial.distance import cdist 

def sinkhorn_distance(X, Y, epsilon, max_iters=1000, stopping_threshold=1e-3):
    # X and Y are the input multidimensional datasets
    # epsilon is the regularization parameter

    # Initialize the Sinkhorn-Knopp algorithm
    n, m = X.shape[0], Y.shape[0]
    K = np.exp(-np.linalg.norm(X[:, np.newaxis] - Y, axis=-1) / epsilon)
    u = np.ones(n) / n
    v = np.ones(m) / m

    # Iteratively update u and v until convergence
    for _ in range(max_iters):
        u_prev = u.copy()
        v_prev = v.copy()

        # Update u
        KtransposeU = K.T.dot(u)
        u = 1.0 / (KtransposeU + 1e-10)

        # Update v
        KU = K.dot(u)
        v = 1.0 / (KU + 1e-10)

        # Check convergence
        if np.sum(np.abs(u - u_prev)) < stopping_threshold and np.sum(np.abs(v - v_prev)) < stopping_threshold:
            break

    # Compute the Sinkhorn distance
    M = np.diag(u).dot(K).dot(np.diag(v))
    row_ind, col_ind = linear_sum_assignment(M)
    sinkhorn_dist = np.sum(M[row_ind, col_ind])

    return sinkhorn_dist





def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()



def plot_mul_models(file_list):
    cur_df = None
    for i, cur_file in enumerate(file_list):
        if 'rank' in cur_file:
            continue
        if i == 0 or cur_df is None:
            cur_df = pd.read_csv(cur_file)
        else:
            cur_df = cur_df.append(pd.read_csv(cur_file))
    _ = cur_df.groupby(['Model', 'Method', 'Amplitude', 'Alpha', 'FDR.nominal']).describe(percentiles=[])
    
    nominal_fdr = 0.1
    signal_amplitude_vec =  [3, 5, 10, 15, 20, 25, 30]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    sns.pointplot(x="Amplitude", y="Power", hue="Method", data=cur_df, ax=ax)
    # fig
    ax = fig.add_subplot(212)
    # fig,ax = plt.subplots(figsize=(12,6))
    ax.plot([-1, np.max(signal_amplitude_vec)+1], 
            [nominal_fdr, nominal_fdr], linewidth=1, linestyle="--", color="red")
    sns.pointplot(x="Amplitude", y="FDP", hue="Method", data=cur_df, ax=ax)
    plt.ylim([0,0.25])
    # fig
    
def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)





def sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(encoded_samples.device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.abs(torch.pow(wasserstein_distance, p))
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()

def sliced_wasserstain_dependency(v1, v2, p=1):
    n = v1.shape[0]
#     v1 = v1.view(n, -1)
#     v2 = v2.view(n, -1)
#     torch.manual_seed(seed)
    shuf = torch.randperm(n)
    xy = torch.cat((v1, v2), dim=1)
    xys = torch.cat((v1[shuf], v2), dim=1)
    return sliced_wasserstein_distance(xy, xys, p=p)


def get_gpu(minimum_mb=2000):
    if torch.cuda.is_available():
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
            logging.debug(f'CUDA_VISIBLE_DEVICES: {visible_devices}')
        else:
            visible_devices = None

        if visible_devices is None:
            logging.debug(f'No CUDA_VISIBLE_DEVICES environment variable...')
            os.system(
                'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > /tmp/gpu-stats'
            )
        else:
            logging.debug(f'CUDA_VISIBLE_DEVICES={visible_devices} found...')
            os.system(
                f'CUDA_VISIBLE_DEVICES="{visible_devices}" nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > /tmp/gpu-stats'
            )
        memory_available = np.array([
            int(x.split()[2]) for x in open('/tmp/gpu-stats', 'r').readlines()
        ])
        os.system('rm -f /tmp/gpu-stats')

        logging.debug(f'Free memory per device: {memory_available}')

        if np.any(memory_available >= minimum_mb):
            logging.debug(f'Using device cuda:{np.argmax(memory_available)}')
            return torch.device(f'cuda:{np.argmax(memory_available)}')
        else:
            logging.debug(f'No free GPU device found. Using cpu...')
            return torch.device('cpu')
    logging.debug(f'No GPU device found. Using cpu...')
    return torch.device('cpu')


def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds


def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b * batch_size:min((b + 1) * batch_size, len(order))]
        yield idx


def logsumexp(inputs, dim=None, keepdim=False, axis=None):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

    Taken from https://github.com/pytorch/pytorch/issues/2591
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.

    if axis is not None:
        dim = axis

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def create_dataloader(*arr, batch_size=64, shuffle=True, drop_last=False):
    """
    Creates pytorch data loaders from numpy arrays
    """
    train_tensors = [torch.from_numpy(a).float() for a in arr]
    train_dataset = torch.utils.data.TensorDataset(*train_tensors)

    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       drop_last=drop_last)


class JointDataset(Dataset):
    def __init__(self, X, Y=None, beta=None, d=None, mode='generation'):
        self.X = X
        self.Y = Y
        self.beta = beta
        self.set_d(d)  # initial stage
        self.mode = mode

        # generation returns only xs
        # prediction returns xs and ys
        assert self.mode in ['generation', 'prediction']

    def __getitem__(self, index):
        if self.mode == 'generation':
            x = self.X[index]
            if self.d is None:
                return tuple([x])
            else:
                if self.d == 0:
                    return tuple([torch.tensor([1.0]), x[0]])
                else:
                    return tuple([x[:self.d], x[self.d]])
        elif self.mode == 'prediction':
            assert self.Y is not None, 'Y cannot be None...'
            x = self.X[index]
            y = self.Y[index]
            return tuple([x, y])
        else:
            raise NotImplementedError(
                f'Data loader mode [{self.mode}] is not implemented...')

    def set_mode(self, mode='generation'):
        """Outputs label data in addition to training data
        x data is all columns of X, y data is Y
        """
        self.mode = mode

    def set_d(self, d=None):
        """Chooses dimension to train on:
            d = 0 -> x data is just 1s, y data is first column of X
            d = 1 -> x data is 1st column of X, y data is second column
            d = 2 -> x data is 1st 2 columns of X, y data is third column

            d must be in the range [0, X.shape[-1] - 1]
        """
        self.d = d

    def reset(self):
        """Resets JointDataset to original state"""
        self.set_mode()
        self.set_d()

    def __len__(self):
        return len(self.X)


def create_jointloader(X,
                       Y=None,
                       beta=None,
                       batch_size=64,
                       shuffle=True,
                       drop_last=False):
    """
    Create pytorch data loader from numpy array.
    Can split data appropriately for complete conditionals
    using the `.set_d()` method.
    Can also split data for prediction tasks for X -> Y
    Stores true feature importance `beta`
    """
    jd = JointDataset(X=X, Y=Y, beta=beta, mode='generation')
    return torch.utils.data.DataLoader(jd,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       drop_last=drop_last)


def get_two_moments(dataloader, safe=True, unbiased=False):
    running_sum = None
    running_counts = None

    for lst in dataloader:
        if running_sum is None:
            tmp_ct = []
            tmp_sum = []
            for elem in lst:
                tmp_ct.append(len(elem))
                tmp_sum.append(elem.sum(axis=0))

            running_sum = tmp_sum
            running_counts = tmp_ct
        else:
            for i, elem in enumerate(lst):
                running_sum[i] += elem.sum(axis=0)
                running_counts[i] += len(elem)

    running_means = [(s / c) for s, c in zip(running_sum, running_counts)]

    # doesn't yet have normalization
    running_var = None
    for lst in dataloader:
        if running_var is None:
            running_var = [((elem - running_means[i])**2).sum(axis=0)
                           for i, elem in enumerate(lst)]
        else:
            for i, elem in enumerate(lst):
                running_var[i] += ((elem - running_means[i])**2).sum(axis=0)

    if unbiased:
        offset = 1
    else:
        offset = 0
    running_stds = [(s / (c - offset))**0.5
                    for s, c in zip(running_var, running_counts)]

    if safe:
        for elem in running_stds:
            elem[elem == 0] = 1.0

    return running_means, running_stds


def extract_data(dataloader):
    out = None

    for lst in dataloader:
        if out is None:
            out = lst
        else:
            tmp = [torch.cat([a, b], axis=0) for a, b in zip(out, lst)]
            out = tmp

    return out


class SafeSoftmax(nn.Module):
    def __init__(self, axis=-1, eps=1e-5):
        """
        Safe softmax class
        """
        super().__init__()
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        """
        apply safe softmax in 
        """

        e_x = torch.exp(x - torch.max(x, axis=self.axis, keepdims=True)[0])
        p = e_x / torch.sum(e_x, axis=self.axis, keepdims=True) + self.eps
        p_sm = p / torch.sum(p, axis=self.axis, keepdims=True)

        return p_sm

class Hyperparameters(object):
    def __init__(self, input_dict):
        for key in input_dict.keys():
            if key not in {'self'}:
                setattr(self, key, input_dict[key])
                if key in {'kw_args', 'kwargs'}:
                    for kw_arg in input_dict[key].keys():
                        setattr(self, kw_arg, input_dict[key][kw_arg])

    def __repr__(self):
        elems = ', '.join([f'{key}={self.__dict__[key]}' for key in self.__dict__.keys()])
        return f'Hyperparameters({elems})'



def generateSamples(distType,p):
    '''
    generate a data sampler based on the given distribution
    '''
    if  distType =="SparseGaussian":
        dataSampler = data.SparseGaussian( p, int(0.3*p))
        
    elif distType == "MultivariateStudentT":
        dataSampler = data.MultivariateStudentT(p, 3, 0.5)
        
    elif distType == "GaussianAR1":
        dataSampler = data.GaussianAR1(p, 0.5)
        
    elif distType== "GaussianMixtureAR1":
#         k = 4
#         prop = (6 + (0.5 + np.arange(k) - k / 2)**2)**0.9
#         prop = prop / prop.sum()
#         rho_list = [0.6**(i + 0.9) for i in range(k)] #0.6 for V_02, 0.5 for v_03
# #         mu_list=[20 * i for i in range(k)]
#         mu_list=[0 * i for i in range(k)]
#         print(mu_list)
#         dataSampler = data.GaussianMixtureAR1(p=p, rho_list=rho_list, mu_list=mu_list, proportions=prop)
        k = 3
        prop = (6 + (0.5 + np.arange(k) - k / 2)**2)**0.9
        prop = prop / prop.sum()
        rho_list = [0.6**(i + 0.9) for i in range(k)] #0.6 for V_02, 0.5 for v_03
    #         mu_list=[20 * i for i in range(k)]
        mu_list=[20 * i for i in range(k)]
        dataSampler = data.GaussianMixtureAR1(p=p, rho_list=rho_list, mu_list=mu_list, proportions=prop)
    elif distType == 'twomoon':
        dataSampler = data.TwoMoon(p)
        
    return dataSampler


def gen_batches(n_samples, batch_size, n_reps):
    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    batches = []
    for rep_id in range(n_reps):
        idx = np.random.permutation(n_samples)
        for i in range(0, math.floor(n_samples/batch_size)*batch_size, batch_size):
            window = np.arange(i,i+batch_size)
            new_batch = idx[window]
            batches += [new_batch]
    return(batches)

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

# generate data Y = X (beta)+ z , z = noise
def sample_Y(X, signal_n=20, signal_a=10.0):
    n,p = X.shape
    beta = np.zeros((p,1))
    beta_nonzero = np.random.choice(p, signal_n, replace=False)
    beta[beta_nonzero,0] = (2*np.random.choice(2,signal_n)-1) * signal_a / np.sqrt(n)
    y = np.dot(X,beta) + np.random.normal(size=(n,1))
    return y,beta


def select(W, beta, nominal_fdr=0.1, offset = 1.0):
    W_threshold = kfilter(W, offset = offset, q=nominal_fdr)
    selected = np.where(W >= W_threshold)[0]
    nonzero = np.where(beta!=0.)[0]
    TP = len(np.intersect1d(selected, nonzero))
    FP = len(selected) - TP
    FDP = FP / max(TP+FP,1.0)
    POW = TP / max(len(nonzero),1.0)
    return selected, FDP, POW
    