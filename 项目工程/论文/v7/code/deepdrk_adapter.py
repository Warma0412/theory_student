"""DeepDRK using the official Transformer, swappers, and SWC objectives.

Research adaptations: 64-wide/two-block architecture for 31 features; portable
device selection; corrected all-coordinate SWC averaging; best-state checkpoint.
No synthetic substitute for the original generative mechanism is used.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

from common import ROOT, aggregate_w, seed_all
from models import module_from_file, lasso_competition, selection

KFT = module_from_file("official_kft", ROOT / "vendor/DeepDRK/KFT.py", ("core_funcs",))
GSW = module_from_file("official_gsw", ROOT / "vendor/DeepDRK/gsw.py")


def individual_dependency(v1, v2, p=2, num_projections=100, max_version=False, gsw_module=None):
    # The released function returns inside its loop, measuring only column zero.
    # Batch the same independent coordinate projections and permutations.
    n, d = v1.shape
    directions = torch.randn(d, 2, num_projections, device=v1.device)
    directions = directions / directions.square().sum(1, keepdim=True).sqrt()
    shuffled = torch.rand(n, d, device=v1.device).argsort(dim=0)
    permuted = v1.gather(0, shuffled)
    original = v1[..., None] * directions[None, :, 0] + v2[..., None] * directions[None, :, 1]
    perturbed = permuted[..., None] * directions[None, :, 0] + v2[..., None] * directions[None, :, 1]
    return (original.sort(0).values - perturbed.sort(0).values).pow(p).abs().mean()


GSW.sliced_wasserstain_dependency_individual = individual_dependency


def train_generator(x, xv, params, seed, prefix):
    seed_all(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    p = x.shape[1]
    model = KFT.KFMAETransformer(input_length=p, num_input_features=1, num_classes=1,
                linear_hidden_size=params.get("width", 64), depth=params.get("layers", 2),
                heads=4, dim_head=16, mlp_dim=128, dropout_p=.1, emd_dropout_p=.1).to(device)
    swappers = [KFT.GumbelSwapper(p, .2).to(device) for _ in range(2)]
    opts = torch.optim.Adam([v for s in swappers for v in s.parameters()], lr=.001)
    optg = torch.optim.AdamW(model.parameters(), lr=params.get("lr", .00001))
    gsw = GSW.GSW(device, swappers, [opts, opts], nofprojections=20)
    gsw.device = device
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    vt = torch.tensor(xv, dtype=torch.float32, device=device)
    best, stale, state, histories = np.inf, 0, None, []
    start = time.monotonic()

    def forward(b):
        return model(b.unsqueeze(-1), torch.zeros((len(b), p + 1), dtype=torch.long, device=device),
                     z=torch.rand((len(b), p, 1), device=device)).reshape(len(b), p)

    for epoch in range(params.get("epochs", 200)):
        losses = []
        model.train()
        for iteration, idx in enumerate(torch.randperm(len(xt), device=device).split(512)):
            if len(idx) < 4:
                continue
            b = xt[idx]
            for s in swappers:
                s.requires_grad_(False)
            model.requires_grad_(True)
            optg.zero_grad()
            k = forward(b)
            swap = torch.stack(gsw.gsw_n_swapper(b, k))
            dep = .5 * (GSW.SWC(b, k, mode="individual") + GSW.SWC(k, b, mode="individual"))
            dep = dep + 5 * (GSW.SWC(b, k, mode="destroy") + GSW.SWC(k, b, mode="destroy"))
            loss = swap.mean() + 2 * dep + 30 * swap.var()
            if not torch.isfinite(loss):
                raise ValueError("DeepDRK nonfinite generator loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optg.step()
            if iteration % 3 == 0:
                model.requires_grad_(False)
                for s in swappers:
                    s.requires_grad_(True)
                opts.zero_grad()
                k = forward(b)
                swap = torch.stack(gsw.gsw_n_swapper(b, k))
                diversity = torch.logsumexp(nn.functional.cosine_similarity(
                    swappers[0].pi_net, swappers[1].pi_net, dim=1), dim=0)
                ls = diversity - swap.mean() - 30 * swap.var()
                ls.backward()
                opts.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                vals = []
                for b in vt.split(512):
                    if len(b) >= 4:
                        k = forward(b)
                        vals.append(float(torch.stack(gsw.gsw_n_swapper(b, k)).mean().cpu()))
            validation = float(np.mean(vals))
            histories.append([epoch + 1, float(np.mean(losses)), validation])
            if validation < best - 1e-5:
                best, stale = validation, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if prefix:
                from common import save_json
                save_json(str(prefix) + f"_seed{seed}_training.json",
                          {"history": histories, "elapsed": time.monotonic() - start})
            if stale >= 6:
                break
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    meta = {"seed": seed, "epochs": epoch + 1, "history": histories,
            "seconds": time.monotonic() - start,
            "parameters": sum(v.numel() for v in model.parameters()),
            "device": str(device), "batch_size": 512,
            "width": params.get("width", 64), "layers": params.get("layers", 2)}
    if prefix:
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "params": params, "metadata": meta}, str(prefix) + f"_seed{seed}.pt")
    return model, meta


def generate(model, x, seed, drp=.5):
    seed_all(seed)
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for b in torch.tensor(x, dtype=torch.float32).split(512):
            b = b.to(device)
            k = model(b.unsqueeze(-1),
                      torch.zeros((len(b), x.shape[1] + 1), dtype=torch.long, device=device),
                      z=torch.rand((len(b), x.shape[1], 1), device=device))
            rows.append(k.reshape(len(b), x.shape[1]).cpu().numpy())
    k = np.concatenate(rows)
    # Published correction: alpha weights the row-permuted original sample.
    return (1 - drp) * k + drp * x[np.random.default_rng(seed).permutation(len(x))]


def diagnostic(x, xk, seed=177):
    n, p = x.shape
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n)
    train, test = np.array_split(ids, 2)
    records = []
    for fraction in (.25, .5, 1.):
        cols = rng.choice(p, max(1, int(p * fraction)), replace=False)
        a = np.column_stack([x, xk])
        b = a.copy()
        b[:, cols], b[:, cols + p] = a[:, cols + p], a[:, cols]
        model = ExtraTreesClassifier(n_estimators=150, min_samples_leaf=8, n_jobs=1, random_state=seed)
        model.fit(np.concatenate([a[train], b[train]]),
                  np.r_[np.zeros(len(train)), np.ones(len(train))])
        pr = model.predict_proba(np.concatenate([a[test], b[test]]))[:, 1]
        target = np.r_[np.zeros(len(test)), np.ones(len(test))]
        auc = float(roc_auc_score(target, pr))
        null = []
        for _ in range(199):
            flip = rng.integers(0, 2, len(test))
            null.append(abs(roc_auc_score(np.r_[flip, 1 - flip], pr) - .5))
        pv = float((1 + np.sum(np.array(null) >= abs(auc - .5))) / 200)
        records.append({"swap_fraction": fraction, "auc": auc, "permutation_p": pv})
    ks = float(np.mean([ks_2samp(x[:, j], xk[:, j]).statistic for j in range(p)]))
    return {"mean_marginal_ks": ks, "classifier_tests": records,
            "empirical_warning": any(abs(r["auc"] - .5) > .1 and r["permutation_p"] < .05 / 3
                                     for r in records),
            "passing_is_not_a_proof": True}


def deepdrk_select(x, y, xv, yv, params, seeds, prefix):
    models, metadata, diagnostics = [], [], []
    for seed in seeds:
        model, meta = train_generator(x, xv, params, seed, prefix)
        models.append(model)
        metadata.append(meta)
        diagnostics.append(diagnostic(xv, generate(model, xv, seed)))
    ys = (y - y.mean()) / max(y.std(), 1e-6)
    ws = []
    for r in range(60):
        model = models[r % len(models)]
        k = generate(model, x, 10000 + seeds[0] + r)
        ws.append(lasso_competition(x, k, ys))
    w = np.array(ws)
    score, native, freq = aggregate_w(w)
    return selection(score, native, {"fits": metadata, "diagnostics": diagnostics,
        "frequency": freq.tolist(), "w": w.tolist(), "draws": 60,
        "adaptations": ["64-wide two-block Transformer", "all-coordinate SWC averaging bug fix",
                        "CPU/MPS port", "best checkpoint saved by X-only swap loss",
                        "60-draw e-value aggregation is a separate downstream extension"]},
        secondary=w.mean(0))
