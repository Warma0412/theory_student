"""VTFS official variational Transformer with documented fixed-budget adaptation.

The released encoder uses epsilon=1 during reparameterization. This adapter keeps
that released behavior; it does not claim an independently verified stochastic VAE.
CUDA-only transfers and broad data imports are removed without changing layers.
The corpus is uniformly sampled subsets (96 utilities), not the optional RL
collector. Thirty-two additional utility queries evaluate decoded candidates.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import ROOT, rank, seed_all
from models import module_from_file, selection, xgb_model

ENC = module_from_file("vtfs_encoder", ROOT / "vendor/VTFS/code/ours/encoder.py",
                       ("feature_env", "utils.logger"))
DEC = module_from_file("vtfs_decoder", ROOT / "vendor/VTFS/code/ours/decoder.py",
                       ("feature_env", "utils.logger"))


class CorpusModel(nn.Module):
    def __init__(self, p, width=64, layers=2):
        super().__init__()
        self.encoder = ENC.TransformerEncoderVAE(
            layers, 4, p + 1, width, .1, "relu", 128, True, 2, width, .1, width
        )
        self.decoder = DEC.TransformerDecoder(layers, 4, p + 1, width, .1,
                                              "relu", 128, True, p, 0)

    def log_prob(self, tokens, memory):
        dec = self.decoder
        embedded = dec.positionalEncoding(dec.embedding(tokens))
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1])
        out = dec.decoder(embedded, memory, tgt_mask=mask)
        return F.log_softmax(dec.out(out), dim=-1)


def vtfs_select(x, y, xv, yv, params, seeds, prefix):
    p = x.shape[1]
    cache = {}
    rng = np.random.default_rng(seeds[0])

    def utility(indices):
        key = tuple(sorted(indices))
        if key not in cache:
            model = xgb_model({"trees": 100, "depth": 3, "lr": .05}, 101).fit(x[:, key], y)
            mse = float(np.mean((model.predict(xv[:, key]) - yv) ** 2))
            cache[key] = -mse
        return cache[key]

    subsets = [tuple(range(p))]
    while len(subsets) < 96:
        k = int(rng.integers(2, p + 1))
        sub = tuple(sorted(rng.choice(p, k, replace=False)))
        if sub not in subsets:
            subsets.append(sub)
    values = np.array([utility(s) for s in subsets])
    low, high = values.min(), values.max()
    targets = (values - low) / max(high - low, 1e-6)
    trained, meta, candidate_orders = [], [], []
    for seed in seeds:
        seed_all(seed)
        model = CorpusModel(p, params.get("width", 64), params.get("layers", 2))
        opt = torch.optim.Adam(model.parameters(), lr=params.get("lr", .001))
        rg = np.random.default_rng(seed)
        history = []
        for epoch in range(params.get("epochs", 100)):
            tokens = np.zeros((len(subsets), p + 1), dtype=np.int64)
            for i, sub in enumerate(subsets):
                tokens[i, :len(sub)] = rg.permutation(np.array(sub) + 1)
            enc = torch.tensor(tokens[:, :p])
            dec_in = torch.cat([torch.zeros((len(subsets), 1), dtype=torch.long), torch.tensor(tokens[:, :p])], 1)
            target = torch.tensor(tokens)
            model.train()
            epoch_losses = []
            for idx in torch.randperm(len(subsets)).split(32):
                memory, _, _, pv, mu, logvar = model.encoder(enc[idx])
                lp = model.log_prob(dec_in[idx], memory)
                reconstruction = F.nll_loss(lp.reshape(-1, p + 1), target[idx].reshape(-1))
                perf = F.mse_loss(pv.flatten(), torch.tensor(targets[idx], dtype=torch.float32))
                kl = -.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())
                loss = .8 * perf + .2 * reconstruction + .001 * kl
                if not torch.isfinite(loss):
                    raise ValueError("VTFS nonfinite training loss")
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                epoch_losses.append(float(loss.detach()))
            history.append(float(np.mean(epoch_losses)))
        model.eval()
        for index in np.argsort(-values)[:4]:
            seq = np.zeros((1, p), dtype=np.int64)
            seq[0, :len(subsets[index])] = np.array(subsets[index]) + 1
            memory, _, _, pv, _, _ = model.encoder(torch.tensor(seq))
            grad, = torch.autograd.grad(pv.sum(), memory)
            memory = F.normalize(memory + grad, p=2, dim=-1).detach()
            tokens = torch.zeros((1, 1), dtype=torch.long)
            order = []
            with torch.no_grad():
                for _ in range(p):
                    lp = model.log_prob(tokens, memory)[0, -1].clone()
                    lp[0] = -torch.inf
                    if order:
                        lp[torch.tensor(order) + 1] = -torch.inf
                    token = int(lp.argmax())
                    order.append(token - 1)
                    tokens = torch.cat([tokens, torch.tensor([[token]])], dim=1)
            candidate_orders.append(order)
        meta.append({"seed": seed, "loss_history": history, "epochs": len(history),
                     "parameters": sum(t.numel() for t in model.parameters())})
        trained.append(model)
        if prefix:
            Path(prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "params": params, "metadata": meta[-1]},
                       str(prefix) + f"_seed{seed}.pt")
    # Choose one complete ordering by the prespecified primary K, not test scores.
    candidate_orders += [list(s) + [j for j in range(p) if j not in s] for s in
                         sorted(subsets, key=utility, reverse=True)[:8]]
    best_order = max(candidate_orders, key=lambda order: utility(order[:12]))
    score = np.empty(p)
    score[best_order] = np.arange(p, 0, -1)
    return selection(score, metadata={
        "fits": meta, "utility_queries": len(cache), "corpus_size": len(subsets),
        "corpus": [{"features": list(s), "negative_validation_mse": v} for s, v in cache.items()],
        "official_reparameterization_epsilon": 1,
        "adaptations": ["CPU device transfers", "uniform subset corpus instead of optional RL collector",
                        "unique-token fixed-K decoding instead of autostop",
                        "primary-K utility used to select a complete ranking"],
    })
