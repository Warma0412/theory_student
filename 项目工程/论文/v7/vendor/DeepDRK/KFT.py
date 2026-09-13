from torch.nn.functional import gelu

import numpy as np

from torch.nn import LSTM
from torch.autograd import Variable
import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from core_funcs import bernoulli_sampling, bernoulli_weights_normalizer, weights_shifter
import math


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

class GumbelSwapper(nn.Module):
    def __init__(self,
                 d,
                 tau,
                 init_frac=[0.5, 0.5],
                 anneal_rate=2e-4,
                 tau_min=0.2,
                 weight_decay=1e-5,
                 adam_betas=(0.9, 0.999), anneal_start_epoch=5):
        # initialize pytoch nn
        super().__init__()

        # initialize hyperparameters
        self.d = d
        self.tau = tau
        self.init_frac = init_frac
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas
        self.anneal_start_epoch = anneal_start_epoch

        # parameters of gumbel softmax
        ## shape = (d, 2)
        init_frac = torch.tensor(init_frac)
        self.pi_net = nn.Parameter(torch.randn([d, 2]) * init_frac, requires_grad=True)#PiNet(self.d, init_frac)

        # define optimizer for parameters
        self.safe_softmax = SafeSoftmax(axis=2)

        # keep track of epoch
        self.epoch = 0

    def update_tau(self):
        """anneal tau"""
        if self.epoch >= self.anneal_start_epoch:
            self.tau = np.maximum(self.tau * np.exp(-self.anneal_rate * self.epoch),
                                self.tau_min)
        self.epoch += 1
        
        
    def reset_parameters(self):
        self.pi_net.data.normal_(0, 1)

    def forward(self, x, x_tilde, perm_mat=False):
        """implements straight-through gumbel softmax
        https://arxiv.org/pdf/1611.01144.pdf (Jang et al. 2017)
        """
        n, d = x.shape
        assert d == self.d# f'Dimension of x ({d}) does not match self.d ({self.d}).'
        assert x.shape == x_tilde.shape, 'x and x_tilde have different dimensions.'

        # sample gumbel softmax
        # logits = F.log_softmax(self.log_pi, dim=1)
        gumbels = -torch.empty(n, d, 2).exponential_().log().type_as(x)
        gumbels = (gumbels + self.pi_net) / self.tau
        y_soft = self.safe_softmax(gumbels)

        # straight through gumbel softmax
        index = y_soft.max(axis=2, keepdim=True)[1]
        y_hard = torch.zeros(n, d, 2).type_as(x).scatter_(2, index, 1.0)
        swaps = y_hard - y_soft.detach() + y_soft

        # create swap matrix
        A = swaps[..., 0]
        B = swaps[..., 1]

        u = B * x + A * x_tilde
        u_tilde = A * x + B * x_tilde
        if perm_mat:
            indices = (A == 1.).nonzero()
            vector1d = torch.arange(2*d).to(x.device)
            vector1d[indices], vector1d[indices+d] = vector1d[indices+d], vector1d[indices]
            perm_mat = torch.eye(2*d).to(x.device)
            perm_mat = perm_mat[vector1d,]
            return u, u_tilde, perm_mat

        # return swapped x and x_tilde
        return u, u_tilde

class LinearProjectionLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_rate=0.1):
        super(LinearProjectionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.lin = nn.Sequential(nn.Linear(self.in_features, self.hidden_features),
                              nn.GELU(), nn.Dropout(dropout_rate),
                              nn.Linear(self.hidden_features, self.out_features),
                              nn.Dropout(dropout_rate))

        self.norm = nn.LayerNorm(self.out_features, eps=1e-6)

    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)

        return x
    
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            mask = mask.float()#mask.to(dots.device).float()
            mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # [B, T, T], which is mask for scaled-dot attention
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)  # B x num_heads x T x T
            mask = mask.masked_fill_(mask == 1., -1e9)
            dots = dots + mask

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            if mask is not None:
                x = attn(x, mask=mask) + x
            else:
                x = attn(x) + x
            x = ff(x) + x
        return x
    
class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, num_input_features, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len + 1  # for class token encoding
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_len, num_input_features)
        position = torch.arange(0., self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., num_input_features, 2) *
                             -(math.log(10000.0) / num_input_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.cls_embedding = nn.Parameter(torch.zeros(1,1,num_input_features), requires_grad=False)
    def forward(self, x, just_embedding=False):
        if not just_embedding:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
            return self.dropout(x) 
        else:
            return Variable(self.pe[:, :], requires_grad=False)

    
class InputConditionPrediction(nn.Module):
    """
    multi-class classification model : some input data arrangement classes
    """

    def __init__(self, hidden_dim, n_class):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden_dim, n_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512, n_class=2):
        super().__init__(n_class, embed_size, padding_idx=0)
        
        
class KFMAETransformer(nn.Module):
    def __init__(self, input_length, num_input_features=2, linear_hidden_size=128,
                 num_classes=2, depth=6, heads=16, mlp_dim=1024,
                 dim_head = 32, dropout_p = 0.1, emd_dropout_p = 0.1):
        super(KFMAETransformer, self).__init__()

        self.num_input_features = num_input_features
        self.linear_hidden_size = linear_hidden_size
        self.dropout_p = dropout_p
        self.emd_dropout_p = emd_dropout_p
        self.num_classes = num_classes
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.zjs = nn.Parameter(torch.zeros(input_length), requires_grad=True)
        self.seg_embedding = SegmentEmbedding(embed_size=self.linear_hidden_size, n_class=self.num_classes)
        self.PreLin = LinearProjectionLayer(in_features=self.num_input_features,
                                      hidden_features=self.linear_hidden_size * 2,
                                      out_features=self.linear_hidden_size,
                                      dropout_rate=self.emd_dropout_p)
        
        self.EncodeZLin = LinearProjectionLayer(in_features=self.linear_hidden_size*2,
                                      hidden_features=self.linear_hidden_size * 2,
                                      out_features=self.linear_hidden_size,
                                      dropout_rate=self.emd_dropout_p)

        self.pos_embedding = PositionalEncoding(num_input_features=self.linear_hidden_size,
                                                dropout=0.0, max_len=input_length*2)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.linear_hidden_size), requires_grad=False)
        self.dropout = nn.Dropout(self.emd_dropout_p)

        self.transformer = Transformer(self.linear_hidden_size, self.depth, self.heads, self.dim_head,
                                       self.mlp_dim, self.dropout_p)
        

        self.input_condition_head = nn.Sequential(
            nn.LayerNorm(self.linear_hidden_size),
            nn.Linear(self.linear_hidden_size, self.num_classes)
        )
        
        self.output_timestep_unify = nn.Sequential(
            nn.LayerNorm(self.linear_hidden_size),
            nn.Linear(self.linear_hidden_size, self.num_input_features)
        )
        
        
        # decoding functions
        decoder_embed_dim = self.linear_hidden_size # can change later on
        self.decoder_embed = nn.Linear(self.linear_hidden_size, decoder_embed_dim, bias=True)
        self.dec_pos_embedding = PositionalEncoding(num_input_features=decoder_embed_dim,
                                                dropout=0.0, max_len=input_length*2)
        self.transformer_decoder = Transformer(self.linear_hidden_size, self.depth, self.heads, self.dim_head,
                                       self.mlp_dim, self.dropout_p)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.output_timestep_unify_decoder = nn.Sequential(
            nn.LayerNorm(self.linear_hidden_size),
            nn.Linear(self.linear_hidden_size, self.num_input_features)
        )
        
        # weights init
        self.initialize_weights()
        self.initialize_zjs()
    
    def initialize_zjs(self):
        torch.nn.init.normal_(self.zjs, std=.2)
        
        
    def initialize_weights(self):
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)



    def forward(self, x, seg_labels, z=None, output_cls=False):
        '''
        x: [batch, seq_length, data_dim]
        seg_labels: [batch, seq_length+1]
        '''
        # linear extentsion on features
        x = self.PreLin(x)
        b, n, fe = x.shape
        if z is not None:
            z = z.repeat([1, 1, fe])
            xz = torch.cat([x, z], dim=2)
            x = self.EncodeZLin(xz)
        
        
        # add cls token and only pos_embedd cls token with zeros
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # class embedding 
        x = x + self.seg_embedding(seg_labels)

        # positional embedding
        x  = self.pos_embedding(x)

        # transformer 
        x = self.transformer(x)
        
        cls_tokens, outputs = x[:, 0, :], x[:, 1:, :]
        outputs = self.output_timestep_unify(outputs).squeeze()
        if output_cls:
            return self.input_condition_head(cls_tokens), outputs
        else:
            return outputs
        
