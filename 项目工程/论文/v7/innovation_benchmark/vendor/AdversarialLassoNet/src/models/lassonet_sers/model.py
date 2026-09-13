from itertools import islice
import torch
from torch import nn
from torch.nn import functional as F
from lassonet.prox import inplace_group_prox, inplace_prox, prox
from itertools import islice

class LassoNet(nn.Module):
    def __init__(self, *dims, groups=None, dropout=None):
        """
        dims: input dimension, hidden layer 1, hidden layer 2, ..., output dimension
        """
        assert len(dims) > 2
        if groups is not None:
            n_inputs = dims[0]
            all_indices = []
            for g in groups:
                for i in g:
                    all_indices.append(i)
            assert len(all_indices) == n_inputs and set(all_indices) == set(
                range(n_inputs)
            ), f"Groups must be a partition of range(n_inputs={n_inputs})"

        self.groups = groups
        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.bn_layers = nn.ModuleList()
        for i in range(len(dims)-2):  
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))

        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

    def forward(self, inp):
        current_layer = inp
        result = self.skip(inp)

    
        for i, theta in enumerate(self.layers):
            current_layer = theta(current_layer)  # Apply the fully connected Linear layer.

            
            if i < len(self.bn_layers):
                current_layer = self.bn_layers[i](current_layer)

            if self.dropout is not None:
                current_layer = self.dropout(current_layer)

        
            current_layer = F.relu(current_layer)

        return result + current_layer

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        if self.groups is None:
            with torch.no_grad():
                inplace_prox(
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )
        else:
            with torch.no_grad():
                inplace_group_prox(
                    groups=self.groups,
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )

    def lambda_start(self, M=1, lambda_bar=0, factor=2):
        def is_sparse(lambda_):
            with torch.no_grad():
                beta = self.skip.weight.data
                theta = self.layers[0].weight.data
                for _ in range(10000):
                    new_beta, theta = prox(
                        beta, theta, lambda_=lambda_, lambda_bar=lambda_bar, M=M
                    )
                    if torch.abs(beta - new_beta).max() < 1e-5:
                        break
                    beta = new_beta
                return (torch.norm(beta, p=2, dim=0) == 0).sum()

        start = 1e-6
        while not is_sparse(factor * start):
            start *= factor
        return start

    def l2_regularization(self):
        ans = 0
        for layer in islice(self.layers, 1, None):
            ans += (torch.norm(layer.weight.data, p=2) ** 2)
        return ans

    def l1_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2, dim=0).sum()

    def l2_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2)

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0

    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}


