import numpy as np
from scipy import stats
import torch
import torch.distributions as dist

def gen_ark_X(n, p, k=2, max_corr=0.99, alpha0=0.2, alphasum=1.0):
	"""
	Generates gaussian X ~ AR(k) and the corresponding cov matrix.
	"""
	alpha = np.zeros(k+1)
	alpha[0] = alpha0
	alpha[1:] = (alphasum - alpha0) / k
	rhos = stats.dirichlet.rvs(alpha, size=p-1) # p-1 x k+1
	# Enforce maximum correlation
	rhos[:, 0] = np.maximum(rhos[:, 0], 1 - max_corr)
	rhos = rhos / rhos.sum(axis=1).reshape(-1, 1) # ensure sums to 1
	rhos = np.sqrt(rhos)

	# Compute eta so that the autoregressive process is equivalent to
	# Z = etas @ W for W i.i.d. standard normals
	etas = np.zeros((p, p))
	etas[0, 0] = 1
	for j in range(1, p):
		# Account for correlations between Xj and X_{1:j-1} 
		rhoend = min(j+1, k+1)
		for i, r in enumerate(np.flip(rhos[j-1,1:rhoend])):
			etas[j] += etas[j-i-1] * r
		# Rescale so Var(Xj) = 1
		scale = np.sqrt((1 - rhos[j-1, 0]**2) / np.power(etas[j], 2).sum())
		etas[j] = etas[j] * scale
		# Add extra noise
		etas[j, j] = rhos[j-1, 0]

	# Ensure data is not constant
	W = np.random.randn(n, p)
	Z = np.dot(W, etas.T)
	V = np.dot(etas, etas.T)
	return Z, V

def sample_beta(p, sparsity, coeff_size, coeff_dist, corr_signals=False, N=None):
	nnn = int(np.ceil(sparsity * p))
	if corr_signals:
		first_nn = np.random.choice(np.arange(p-nnn+1))
		nnull_inds = np.arange(first_nn, first_nn+nnn)
	else:
		nnull_inds = np.random.choice(
			np.arange(p), nnn, replace=False
		)
	beta = np.zeros(p)
	signs = 2 * np.random.binomial(1, 0.5, size=nnn) - 1
	if coeff_dist == 'normal':
		nnull_vals = np.sqrt(coeff_size) * np.random.randn(nnn)
	elif coeff_dist == 'fixed':
		nnull_vals = 1. if p == 100 else 1. #the value is the signal amplitude
	elif coeff_dist == 'uniform':
		nnull_vals = np.random.uniform(low=coeff_size/2, high=coeff_size, size=nnn)
	elif coeff_dist == 'scale':
		nnull_vals = float(p) / np.sqrt(N) / 15.
	elif coeff_dist == 'scale125':
		nnull_vals = float(p) / np.sqrt(N) / 12.5
	elif coeff_dist == 'expo':
		nnull_vals = np.sqrt(coeff_size) * stats.expon.rvs(size=nnn)
	elif coeff_dist == 't':
		nnull_vals = np.sqrt(coeff_size) * stats.t.rvs(size=nnn, df=3)
	elif coeff_dist == 'mixture':
		# large mixture
		nnull_vals = np.sqrt(coeff_size) * np.random.randn(nnn)
		small_mixture = np.random.binomial(1, 0.5, size=nnn).astype(bool)
		nsm = small_mixture.sum()
		nnull_vals[small_mixture] =  np.sqrt(coeff_size) / 10 * np.random.randn(nsm)
	else:
		raise ValueError(f"Unrecognized coeff_dist={coeff_dist}")
	beta[nnull_inds] = signs * nnull_vals
	return beta

def sample_y(mu, y_dist):

	# Generate y | X
	n = mu.shape[0]
	if y_dist == 'gaussian':
		y = mu + np.random.randn(n)
	elif y_dist == 'probit':
		y = mu + np.random.randn(n)
		y = (y < 0).astype(int)
	elif y_dist == 'binomial' or y_dist == 'logistic':
		probs = np.exp(mu) / (1.0 + np.exp(mu))
		y = np.random.binomial(1, probs).astype(int)
	elif y_dist == 'cauchy':
		y = mu + stats.cauchy.rvs(size=n)
		y = y / y.std()
	elif y_dist == 'laplace':
		y = mu + stats.laplace.rvs(size=n)
	elif y_dist == 'expo':
		y = mu + stats.expon.rvs(size=n)
		y = (y - y.mean()) / y.std()
	else:
		raise ValueError(f"Unrecognized y_dist={y_dist}")
	return y



def sample_beta_torch(p, sparsity, coeff_size, coeff_dist, device, corr_signals=False):
    nnn = int(np.ceil(sparsity * p))#.to(device)
    if corr_signals:
        first_nn = torch.randint(high=p - nnn + 1, size=(1,)).item()
        nnull_inds = torch.arange(first_nn, first_nn + nnn).to(device)
    else:
        nnull_inds = torch.randperm(p)[:nnn].to(device)

    beta = torch.zeros(p).to(device)
    signs = 2 * torch.randint(0, 2, size=(nnn,)).to(device) - 1

    if coeff_dist == 'normal':
        nnull_vals = torch.sqrt(coeff_size) * torch.randn(nnn)
    elif coeff_dist == 'uniform':
        nnull_vals = torch.rand(nnn) * (coeff_size - coeff_size / 2) + coeff_size / 2
    elif coeff_dist == 'expo':
        nnull_vals = torch.sqrt(coeff_size) * torch.tensor(stats.expon.rvs(size=nnn)).float()
    elif coeff_dist == 't':
        nnull_vals = torch.sqrt(coeff_size) * torch.tensor(stats.t.rvs(size=nnn, df=3)).float()
    elif coeff_dist == 'mixture':
        nnull_vals = torch.sqrt(coeff_size) * torch.randn(nnn)
        small_mixture = torch.bernoulli(0.5 * torch.ones(nnn)).bool()
        nsm = small_mixture.sum()
        nnull_vals[small_mixture] = torch.sqrt(coeff_size) / 10 * torch.randn(nsm)
    else:
        raise ValueError(f"Unrecognized coeff_dist={coeff_dist}")

    beta[nnull_inds] = signs * nnull_vals.to(device)
    return beta



def sample_y_torch(mu, y_dist, device):
    # Generate y | X
    n = mu.shape[0]
    
    if y_dist == 'gaussian':
        y = mu + torch.randn(n).to(device)
    elif y_dist == 'probit':
        y = mu + torch.randn(n).to(device)
        y = (y < 0).type(torch.int)
    elif y_dist == 'binomial' or y_dist == 'logistic':
        probs = torch.exp(mu) / (1.0 + torch.exp(mu))
        y = dist.Bernoulli(probs).sample().type(torch.int)
    elif y_dist == 'cauchy':
        y = mu + torch.tensor(stats.cauchy.rvs(size=n)).float().to(device)
        y = y / y.std()
    elif y_dist == 'laplace':
        y = mu + torch.tensor(stats.laplace.rvs(size=n)).float().to(device)
    elif y_dist == 'expo':
        y = mu + torch.tensor(stats.expon.rvs(size=n)).float().to(device)
        y = (y - y.mean()) / y.std()
    else:
        raise ValueError(f"Unrecognized y_dist={y_dist}")
        
    return y