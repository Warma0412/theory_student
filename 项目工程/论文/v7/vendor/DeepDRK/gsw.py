import numpy as np

import torch
from torch import optim
import torch.nn as nn

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
                                 num_projections=100,
                                 p=2):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            xxx device (torch.device): torch device (default 'cpu')

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


def sliced_wasserstain_dependency_multi(v1, v2, p=2, num_projections=100, max_version=False, gsw_module=None, check_ratio=0.1):
    n1, d1 = v1.shape
    n2, d2 = v2.shape
    assert n1 == n2 # right now focus on the same sample size case
    num_select1 = int(d1*check_ratio)
    num_select2 = int(d2*check_ratio)
    idx_set1 = np.random.choice(range(d1), num_select1, replace=False)
    idx_set2 = np.random.choice(range(d2), num_select2, replace=False)
    shuf1 = torch.randperm(n1)
    xy1 = torch.cat((v1[..., idx_set1], v2), dim=1)
    xys1 = torch.cat((v1[shuf1, ...][..., idx_set1], v2), dim=1)
    shuf2 = torch.randperm(n2)
    xy2 = torch.cat((v1, v2[..., idx_set2]), dim=1)
    xys2 = torch.cat((v1, v2[shuf2, ...][..., idx_set2]), dim=1)
    
    if not max_version:
        return sliced_wasserstein_distance(xy1, xys1, p=p, num_projections=num_projections) + sliced_wasserstein_distance(xy2, xys2, p=p, num_projections=num_projections)
    else:
        assert gsw_module is not None
        return gsw_module.max_gsw(xy1, xys1, iterations=5,lr=1e-4) + gsw_module.max_gsw(xy2, xys2, iterations=5,lr=1e-4)
    
    
def sliced_wasserstain_dependency_n_perm(v1, v2, p=2, num_projections=100, max_version=False, gsw_module=None, n_perm=10):
    out_loss_list = []
    for _ in range(n_perm):
        out_loss_list.append(sliced_wasserstain_dependency(v1, v2, p=p, num_projections=num_projections,
                                      max_version=max_version, gsw_module=gsw_module))
    return torch.mean(torch.stack(out_loss_list))
        
        
    
def sliced_wasserstain_dependency(v1, v2, p=2, num_projections=100, max_version=False, gsw_module=None):
    n = v1.shape[0]
    shuf = torch.randperm(n)
    xy = torch.cat((v1, v2), dim=1)
    xys = torch.cat((v1[shuf], v2), dim=1)
    if not max_version:
        return sliced_wasserstein_distance(xy, xys, p=p, num_projections=num_projections)
    else:
        assert gsw_module is not None
        return gsw_module.max_gsw(xy, xys, iterations=5,lr=1e-4)
    
def sliced_wasserstain_dependency_individual(v1, v2, p=2, num_projections=100, max_version=False, gsw_module=None):
    n, dims = v1.shape
    out_list = []
    for d in range(dims):
        shuf = torch.randperm(n)
        tmp_v1 = v1[:, d][..., None]
        tmp_v2 = v2[:, d][..., None]
        xy = torch.cat((tmp_v1, tmp_v2), dim=1)
        xys = torch.cat((tmp_v1[shuf], tmp_v2), dim=1)
        if not max_version:
            out = sliced_wasserstein_distance(xy, xys, p=p, num_projections=num_projections)
            out_list.append(out)
        else:
            assert gsw_module is not None
            out = gsw_module.max_gsw(xy, xys, iterations=5,lr=1e-4)
            out_list.append(out)
        return torch.stack(out_list).mean()
    

    
def sliced_wasserstain_dependency_1vsOther(v1, v2, p=2, num_projections=100, max_version=False, gsw_module=None):
    n, dims = v1.shape
    out_list = []
    for d in range(dims):
        shuf = torch.randperm(n)
        tmp_v1 = v1[:, d][..., None]
        other_indices = [d_i for d_i in range(dims) if d_i != d]
        tmp_v2 = v2[:, other_indices]
        xy = torch.cat((tmp_v1, tmp_v2), dim=1)
        xys = torch.cat((tmp_v1[shuf], tmp_v2), dim=1)
        if not max_version:
            out = sliced_wasserstein_distance(xy, xys, p=p, num_projections=num_projections)
            out_list.append(out)
        else:
            assert gsw_module is not None
            out = gsw_module.max_gsw(xy, xys, iterations=5,lr=1e-4)
            out_list.append(out)
        return torch.stack(out_list).mean()
    
def sliced_wasserstain_dependency_KvsOther(v1, v2, p=2, num_projections=20, max_version=False, gsw_module=None, k=10):
    n, dims = v1.shape
    assert k < dims
    out_list = []
    for d in range(dims):
        shuf = torch.randperm(n)
        
        other_indices = [d_i for d_i in range(dims) if d_i != d]
        random_subset = np.random.choice(other_indices, k-1, replace=False)
        tmp_v1 = torch.cat([v1[:, d][..., None], v1[:, random_subset]], dim=1)
        tmp_v2 = v2[:, other_indices]
        xy = torch.cat((tmp_v1, tmp_v2), dim=1)
        xys = torch.cat((tmp_v1[shuf], tmp_v2), dim=1)
        if not max_version:
            out = sliced_wasserstein_distance(xy, xys, p=p, num_projections=num_projections)
            out_list.append(out)
        else:
            assert gsw_module is not None
            out = gsw_module.max_gsw(xy, xys, iterations=5,lr=1e-4)
            out_list.append(out)
        return torch.stack(out_list).mean()
    
def SWC(v1, v2, p=2, num_projections=100, max_version=False, gsw_module=None, mode='individual', k=6, correlation=True):
    '''
    sliced wasserstein correlation
    mode: ['destroy', '1vsOther', 'individual', 'KvsOther', 'None']
    '''
    if mode == 'destroy':
        numerator = sliced_wasserstain_dependency(v1, v2, p, num_projections, max_version, gsw_module)
        denominator = torch.sqrt(sliced_wasserstain_dependency(v1, v1, p, num_projections, max_version, gsw_module) * sliced_wasserstain_dependency(v2, v2, p, num_projections, max_version, gsw_module))
    if mode == '1vsOther':
        numerator = sliced_wasserstain_dependency_1vsOther(v1, v2, p, num_projections, max_version, gsw_module)
        denominator = torch.sqrt(sliced_wasserstain_dependency_1vsOther(v1, v1, p, num_projections, max_version, gsw_module) * sliced_wasserstain_dependency_1vsOther(v2, v2, p, num_projections, max_version, gsw_module))  
    if mode == 'KvsOther':
        numerator = sliced_wasserstain_dependency_KvsOther(v1, v2, p, num_projections, max_version, gsw_module, k=k)
        denominator = torch.sqrt(sliced_wasserstain_dependency_KvsOther(v1, v1, p, num_projections, max_version, gsw_module, k=k) * sliced_wasserstain_dependency_KvsOther(v2, v2, p, num_projections, max_version, gsw_module, k=k))
    
    if mode == 'individual':
        numerator = sliced_wasserstain_dependency_individual(v1, v2, p, num_projections, max_version, gsw_module)
        denominator = torch.sqrt(sliced_wasserstain_dependency_individual(v1, v1, p, num_projections, max_version, gsw_module) * sliced_wasserstain_dependency_individual(v2, v2, p, num_projections, max_version, gsw_module))
    
    if mode == 'None':
        numerator = 0.
        denominator = 1.
    if correlation:
        return numerator/(denominator + 1e-7)
    else:
        return numerator
    
    
def sliced_wasserstain_distance_individual(v1, v2, p=2, num_projections=100, gsw_module=None):
    n, dims = v1.shape
    out_list = []
    for d in range(dims):
        tmp_v1 = v1[:, d][..., None]
        tmp_v2 = v2[:, d][..., None]
        out = gsw_module.gsw(tmp_v1, tmp_v2)
        out_list.append(out)
    return torch.stack(out_list).mean()
        
    
def reset_parameters(module):
    if isinstance(module, nn.Module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.reset_parameters()

               
                
class GSW():
    def __init__(self, device, swappers_list=None, swappers_opts_list=None,
                 ftype='linear',nofprojections=10,degree=2,radius=2.,use_cuda=True):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.degree=degree
        self.radius=radius
        self.swappers_list = swappers_list
        self.swappers_opts_list = swappers_opts_list
        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device(device)
        else:
            self.device=torch.device('cpu')
        self.theta=None # This is for max-GSW
    
    
    def max_loss_given_swapper(self, xb, xb_tilde, swapper, loss_cal, step=1):
        check = None
        for _ in range(step):
            xb, xb_tilde = swapper(xb, xb_tilde)
            check_tmp = loss_cal(xb, xb_tilde).mean()
            if not check or check_tmp > check: #confirmed
                check = check_tmp
                out_x, out_xtilde = xb, xb_tilde
        return out_x, out_xtilde, check 

    def gsw(self,X,Y,theta=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        return torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2))
    
    def reset_swappers(self):
        for cur_swapper in self.swappers_list:
            cur_swapper.reset_parameters()
            
            
    def gsw_n_swapper(self, X, Y, n_swapper=10, opt_swapper=False):
        loss_out = []
        for cur_swapper, cur_swapper_opt in zip(self.swappers_list, self.swappers_opts_list):
            
            if opt_swapper:
                self.max_swappers(X, Y, cur_swapper, cur_swapper_opt, iterations=n_swapper,
                                  theta=None)
            
            X_, Y_ = cur_swapper(X, Y)
            loss_out.append(self.gsw(torch.cat([X, Y], dim=1), torch.cat([X_, Y_], dim=1), None))
        return loss_out
            
            
    def max_gsw_n_swapper_losses(self, X, Y, n_gsw=3, n_swapper=1, lr_gsw=3e-4, opt_swapper=False):
        #self.reset_swappers()
        loss_out = []
        for cur_swapper, cur_swapper_opt in zip(self.swappers_list, self.swappers_opts_list):
            
                
            X_, Y_ = cur_swapper(X, Y)
            cur_theta = self.max_gsw_get_theta(torch.cat([X, Y], dim=1), torch.cat([X_, Y_], dim=1),
                                   iterations=n_gsw, lr=lr_gsw, reset=True)
            
            
            if opt_swapper:
                self.max_swappers(X, Y, cur_swapper, cur_swapper_opt, iterations=n_swapper, theta=cur_theta)
                X_, Y_ = cur_swapper(X, Y)

            loss_out.append(self.gsw(torch.cat([X, Y], dim=1), torch.cat([X_, Y_], dim=1), cur_theta))
            

        return loss_out
                
    def max_swappers(self, X, Y, swapper, swapper_opt, iterations, theta=None):
        for i in range(iterations):
            swapper_opt.zero_grad()
            X_, Y_ = swapper(X, Y)
            loss=-self.gsw(torch.cat([X, Y], dim=1), torch.cat([X_, Y_], dim=1), self.theta if theta is None else theta)
            loss.backward(retain_graph=True)
            swapper_opt.step() 
        return
            
        
    def max_gsw_get_theta(self,X,Y,iterations=50,lr=1e-4, reset=True):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        assert dn==dm and M==N
#         if self.theta is None:
        if reset:
            if self.ftype=='linear':
                theta=torch.randn((1,dn),device=device,requires_grad=True)
                theta.data/=torch.sqrt(torch.sum((theta.data)**2))
            elif self.ftype=='poly':
                dpoly=self.homopoly(dn,self.degree)
                theta=torch.randn((1,dpoly),device=device,requires_grad=True)
                theta.data/=torch.sqrt(torch.sum((theta.data)**2))
            elif self.ftype=='circular':
                theta=torch.randn((1,dn),device=device,requires_grad=True)
                theta.data/=torch.sqrt(torch.sum((theta.data)**2))
                theta.data*=self.radius
            self.theta=theta
#             iterations=1000
#         else:
#             iterations=iterations
        optimizer=optim.Adam([self.theta],lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.sqrt(torch.sum(self.theta.data**2))

        self.theta.to(self.device)
        return self.theta.to(self.device)

    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        self.max_gsw_get_theta(X,Y,iterations=iterations,lr=lr)
        return self.gsw(X.to(self.device),Y.to(self.device),
                  self.theta)

    def gsl2(self,X,Y,theta=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Yslices_sorted=torch.sort(Yslices,dim=0)

        return torch.sqrt(torch.sum((Xslices-Yslices)**2))

    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if self.ftype=='linear':
            return self.linear(X,theta)
        elif self.ftype=='poly':
            return self.poly(X,theta)
        elif self.ftype=='circular':
            return self.circular(X,theta)
        else:
            raise Exception('Defining function not implemented')

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='poly':
            dpoly=self.homopoly(dim,self.degree)
            theta=torch.randn((self.nofprojections,dpoly))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='circular':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([self.radius*th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())

    def poly(self,X,theta):
        ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial
        '''
        N,d=X.shape
        assert theta.shape[1]==self.homopoly(d,self.degree)
        powers=list(self.get_powers(d,self.degree))
        HX=torch.ones((N,len(powers))).to(self.device)
        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=X[:,i]**p
        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())

    def circular(self,X,theta):
        ''' The circular defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
        '''
        N,d=X.shape
        if len(theta.shape)==1:
            return torch.sqrt(torch.sum((X-theta)**2,dim=1))
        else:
            return torch.stack([torch.sqrt(torch.sum((X-th)**2,dim=1)) for th in theta],1)

    def get_powers(self,dim,degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation

    def homopoly(self,dim,degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return len(list(self.get_powers(dim,degree)))