"""Local mechanism variants, not claimed original published algorithms."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
V7 = STUDY.parent
sys.path.insert(0, str(V7 / "code"))
from common import seed_all, rank
from models import fit_neural, deep_lasso_scores, xgb_model

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tabm import TabM


def result(score, metadata=None, selected=None):
    order = rank(score)
    if selected is not None:
        chosen = set(map(int, selected))
        order = np.r_[[j for j in order if j in chosen], [j for j in order if j not in chosen]]
    return {"score": np.asarray(score), "order": order, "metadata": metadata or {}}


def mlp(p, params):
    layers = []
    for _ in range(params.get("layers", 2)):
        layers += [nn.Linear(p, params["width"]), nn.ReLU(), nn.Dropout(.1)]
        p = params["width"]
    return nn.Sequential(*layers, nn.Linear(p, 1))


def jacobian_scores(model, x):
    squared = np.zeros(x.shape[1])
    model.eval()
    for batch in torch.as_tensor(x, dtype=torch.float32).split(512):
        xb = batch.clone().requires_grad_(True)
        gradient, = torch.autograd.grad(model(xb).sum(), xb)
        squared += gradient.detach().square().sum(0).numpy()
    return np.sqrt(squared / len(x))


def fit_jacobian(x, y, xv, yv, params, penalty, seed):
    seed_all(seed)
    model = mlp(x.shape[1], params)
    mean, std = float(y.mean()), max(float(y.std()), 1e-6)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor((y - mean) / std, dtype=torch.float32)
    xvt = torch.tensor(xv, dtype=torch.float32)
    yvt = torch.tensor((yv - mean) / std, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=2e-4)
    best, stale, state, history = np.inf, 0, None, []
    for epoch in range(100):
        model.train()
        rows = []
        for idx in torch.randperm(len(x)).split(512):
            xb = xt[idx].clone().requires_grad_(True)
            prediction = model(xb).squeeze(-1)
            mse = (prediction - yt[idx]).square().mean()
            gradient, = torch.autograd.grad(prediction.sum(), xb, create_graph=True)
            sparse = gradient.square().mean(0).add(1e-8).sqrt().sum()
            loss = mse + penalty * sparse
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            rows.append([float(mse.detach()), float(sparse.detach())])
        model.eval()
        with torch.no_grad():
            validation = float((model(xvt).squeeze(-1) - yvt).square().mean())
        if not np.isfinite(validation):
            raise ValueError("Nonfinite Jacobian validation loss")
        history.append([epoch + 1, *np.mean(rows, 0).tolist(), validation])
        if validation < best - 1e-6:
            best, stale, state = validation, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(state)
    model.eval()
    model.y_mean, model.y_std = mean, std
    return model, {"seed": seed, "penalty": penalty, "history": history,
                   "epochs": len(history), "parameters": sum(p.numel() for p in model.parameters())}


def deep_family(x, y, xv, yv, params, penalty, seeds, save_prefix=None):
    jacobian, lossrank, score_only, histories = [], [], [], []
    for seed in seeds:
        model, meta = fit_jacobian(x, y, xv, yv, params, penalty, seed)
        jacobian.append(jacobian_scores(model, xv))
        lossrank.append(deep_lasso_scores(model, xv, yv))
        if save_prefix:
            torch.save({"state_dict": model.state_dict(), "params": params,
                        "y_mean": model.y_mean, "y_std": model.y_std, "meta": meta},
                       str(save_prefix) + f"_jac_seed{seed}.pt")
        original, base_meta = fit_neural(x, y, xv, yv, params, "deep_lasso", seed)
        score_only.append(jacobian_scores(original, xv))
        histories.append(meta | {"original_epochs": base_meta["epochs"]})
    return {
        "dl_jacobian": result(np.mean(jacobian, 0), {"fits": histories}),
        "dl_jacobian_lossrank": result(np.mean(lossrank, 0), {"fits": histories}),
        "dl_score_only": result(np.mean(score_only, 0), {"original_fit_seeds": list(seeds)}),
    }


def soft_cardinality(logits, k, temperature):
    # Implicit derivative of the sigmoid threshold enforces sum(g)=K.
    with torch.no_grad():
        left, right = logits.min() - 30 * temperature, logits.max() + 30 * temperature
        for _ in range(40):
            middle = (left + right) / 2
            if torch.sigmoid((logits - middle) / temperature).sum() > k:
                left = middle
            else:
                right = middle
        boundary = (left + right) / 2
        prob = torch.sigmoid((logits - boundary) / temperature)
        weight = prob * (1 - prob)
    correction = (logits * weight).sum() / weight.sum().clamp_min(1e-8)
    threshold = boundary + correction - correction.detach()
    return torch.sigmoid((logits - threshold) / temperature)


class BudgetTabM(nn.Module):
    def __init__(self, p, k, params):
        super().__init__()
        self.k = k
        self.logits = nn.Parameter(torch.randn(p) * .001)
        self.backbone = TabM.make(n_num_features=p, d_out=1, k=8,
                                 n_blocks=params["layers"], d_block=params["width"], dropout=.1)
        self.temperature = 1.0
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            gate = torch.ones_like(self.logits)
        else:
            hard = torch.zeros_like(self.logits).scatter(0, self.logits.topk(self.k).indices, 1.0)
            soft = soft_cardinality(self.logits, self.k, self.temperature)
            gate = hard + soft - soft.detach() if self.training else hard
        return self.backbone(x * gate)


def gated_family(x, y, xv, yv, params, gate_lr, k, seeds, save_prefix=None):
    scores, histories = [], []
    for seed in seeds:
        seed_all(seed)
        model = BudgetTabM(x.shape[1], k, params)
        mean, std = float(y.mean()), max(float(y.std()), 1e-6)
        xt, xvt = torch.tensor(x), torch.tensor(xv)
        yt = torch.tensor((y - mean) / std, dtype=torch.float32)
        yvt = torch.tensor((yv - mean) / std, dtype=torch.float32)
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": params["lr"], "weight_decay": 2e-4},
            {"params": [model.logits], "lr": gate_lr, "weight_decay": 0.0},
        ])
        best, stale, state, history = np.inf, 0, None, []
        for epoch in range(100):
            model.train()
            model.warmup = epoch < 15
            model.temperature = max(.2, 1.0 - .8 * max(0, epoch - 15) / 85)
            for idx in torch.randperm(len(xt)).split(512):
                output = model(xt[idx]).squeeze(-1)
                loss = (output - yt[idx, None]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                prediction = torch.cat([model(b) for b in xvt.split(1024)]).mean(1).squeeze(-1)
                validation = float((prediction - yvt).square().mean())
            if not np.isfinite(validation):
                raise ValueError("Nonfinite budget-gate loss")
            history.append([epoch + 1, validation])
            if epoch >= 25:
                if validation < best - 1e-6:
                    best, stale, state = validation, 0, copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if stale >= 12:
                    break
        model.load_state_dict(state)
        logits = model.logits.detach().numpy()
        # Rank aggregation is scale invariant across independent gate fits.
        per_seed_score = np.empty(len(logits))
        per_seed_score[rank(logits)] = np.arange(len(logits), 0, -1)
        scores.append(per_seed_score)
        meta = {"seed": seed, "gate_lr": gate_lr, "epochs": len(history),
                "history": history, "logits": logits.tolist(),
                "parameters": sum(p.numel() for p in model.parameters())}
        histories.append(meta)
        if save_prefix:
            torch.save({"state_dict": model.state_dict(), "params": params, "k": k,
                        "y_mean": mean, "y_std": std, "meta": meta},
                       str(save_prefix) + f"_gate_k{k}_seed{seed}.pt")
    return {"tabm_budget_gate": result(np.mean(scores, 0), {"fits": histories, "trained_k": k})}


ENTRY_PATH = V7 / "innovation_benchmark/vendor/EntryPrune/entryprune/code.py"
spec = importlib.util.spec_from_file_location("entryprune_pinned", ENTRY_PATH)
EP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EP)


class EPRegression(EP.BaseEP):
    def __init__(self, p, n, k):
        self.n_vars, self.n_train = p, n
        super().__init__(relevant=k, epochs=100, n_hidden=64, lr=.003,
                         batch_size=512, candidate_perc=.5, switch_interval=5,
                         n_updates=60, shrink=False)
        self.device = torch.device("cpu")
        self.model = nn.Sequential(nn.Linear(self.netsize, 64), nn.ReLU(), nn.Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.003)
        self.criterion = nn.MSELoss()
        self.selected = torch.tensor(np.random.choice(p, self.netsize, replace=False))
        self.ident_index = torch.tensor(np.random.choice(self.netsize, k, replace=False))
        self.ident = self.selected[self.ident_index]
        self.selected_candidate_mask = torch.ones(self.netsize, dtype=torch.bool)
        self.highscores = torch.full((p,), -100.0)
        self.patience_counter_shrink = 0


def ep_one(x, y, xv, yv, env, k, mode, dispersion, seed):
    seed_all(seed)
    learner = EPRegression(x.shape[1], len(x), k)
    mean, std = float(y.mean()), max(float(y.std()), 1e-6)
    xt, xvt = torch.tensor(x), torch.tensor(xv)
    yt = torch.tensor((y - mean) / std, dtype=torch.float32)[:, None]
    yvt = torch.tensor((yv - mean) / std, dtype=torch.float32)[:, None]
    probe_rng = np.random.default_rng(seed + 905)
    probes = []
    for group in np.unique(env):
        idx = np.flatnonzero(env == group)
        probes.append(torch.tensor(probe_rng.choice(idx, min(len(idx), 128), replace=False)))
    history, best, snapshot = [], np.inf, None
    counts = np.zeros(x.shape[1])
    iterator = iter([])
    for update in range(60):
        total_gradient = torch.zeros_like(learner.model[0].weight)
        for _ in range(5):
            try:
                idx = next(iterator)
            except StopIteration:
                iterator = iter(torch.randperm(len(x)).split(512))
                idx = next(iterator)
            prediction = learner.model(xt[idx][:, learner.selected])
            loss = (prediction - yt[idx]).square().mean()
            learner.optimizer.zero_grad()
            loss.backward()
            total_gradient += learner.model[0].weight.grad.detach() * len(idx)
            learner.optimizer.step()
        if mode == "ep_regression":
            learner.update_network(total_gradient, 0, update)
        else:
            probe_scores = []
            for idx in probes:
                out = learner.model(xt[idx][:, learner.selected])
                ploss = (out - yt[idx]).square().mean()
                grad, = torch.autograd.grad(ploss, learner.model[0].weight)
                score = grad.abs().sum(0)
                score = (score - score.mean()) / score.std().clamp_min(1e-8)
                probe_scores.append(score)
            matrix = torch.stack(probe_scores)
            score = matrix.mean(0)
            if mode == "ep_dispersion":
                score = score - dispersion * matrix.std(0, unbiased=False)
            learner.highscores[learner.selected] = score.detach()
            learner.ident_index = torch.topk(score, k).indices
            learner.ident = learner.selected[learner.ident_index].clone()
            candidates = torch.ones(learner.netsize, dtype=torch.bool)
            candidates[learner.ident_index] = False
            pool = [j for j in range(x.shape[1]) if j not in learner.ident.tolist()]
            learner.selected[candidates] = torch.tensor(np.random.choice(pool, int(candidates.sum()), replace=False))
            with torch.no_grad():
                learner.model[0].weight[:, candidates] = torch.empty(int(candidates.sum())).uniform_(-1e-8, 1e-8)
            # Unlike the published baseline, renewed features do not inherit Adam momentum.
            state = learner.optimizer.state.get(learner.model[0].weight, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][:, candidates] = 0
        counts[learner.ident.tolist()] += 1
        with torch.no_grad():
            masked = xvt[:, learner.selected].clone()
            keep = torch.isin(learner.selected, learner.ident)
            masked[:, ~keep] = 0
            validation = float((learner.model(masked) - yvt).square().mean())
        history.append([update + 1, validation, learner.ident.tolist()])
        if not np.isfinite(validation):
            raise ValueError("Nonfinite EntryPrune validation loss")
        if validation < best:
            best = validation
            snapshot = {"state_dict": copy.deepcopy(learner.model.state_dict()),
                        "selected": learner.selected.clone(), "ident": learner.ident.clone(),
                        "best_update": update + 1}
    support = np.zeros(x.shape[1])
    support[snapshot["ident"].tolist()] = 1.0
    # Consensus membership dominates trajectory frequency; fixed K is retained.
    scores = support + .1 * counts / 60
    meta = {"seed": seed, "history": history, "best_update": snapshot["best_update"],
            "dispersion": dispersion if mode == "ep_dispersion" else 0,
            "probe_environments": len(probes), "trained_k": k,
            "optimizer_reset_is_shared_by_refresh_and_dispersion": mode != "ep_regression"}
    snapshot.update({"y_mean": mean, "y_std": std, "meta": meta})
    return scores, meta, snapshot


def entry_family(x, y, xv, yv, env, k, dispersion, seeds, save_prefix=None):
    results = {}
    for mode in ("ep_regression", "ep_refresh", "ep_dispersion"):
        scores, fits = [], []
        for seed in seeds:
            score, meta, snapshot = ep_one(x, y, xv, yv, env, k, mode, dispersion, seed)
            scores.append(score)
            fits.append(meta)
            if save_prefix:
                torch.save(snapshot, str(save_prefix) + f"_{mode}_k{k}_seed{seed}.pt")
        results[mode] = result(np.mean(scores, 0), {"fits": fits, "trained_k": k})
    return results


def subset_family(x, y, xv, yv, k, seeds, save_prefix=None):
    from vtfs_adapter import CorpusModel
    p = x.shape[1]
    rng = np.random.default_rng(seeds[0])
    pool = []
    while len(pool) < 256:
        subset = tuple(sorted(rng.choice(p, k, replace=False).tolist()))
        if subset not in pool:
            pool.append(subset)
    cache = {}

    def utility(subset):
        subset = tuple(sorted(subset))
        if subset not in cache:
            model = xgb_model({"trees": 100, "depth": 3, "lr": .05}, 101).fit(x[:, subset], y)
            cache[subset] = -float(np.mean((model.predict(xv[:, subset]) - yv) ** 2))
        return cache[subset]

    random_values = [utility(s) for s in pool[:128]]
    best_random = pool[int(np.argmax(random_values))]
    scores = np.zeros(p)
    scores[list(best_random)] = 1
    output = {"random_k_search": result(scores, {"queries": 128, "trained_k": k,
                                                 "best_utility": max(random_values)},
                                        selected=best_random)}
    values = np.asarray([utility(s) for s in pool[:96]])
    targets = (values - values.min()) / max(np.ptp(values), 1e-6)
    for mode, pairweight in (("vtfs_cardinality", 0.0), ("vtfs_pairrank", .1)):
        candidate_orders, fits = [], []
        for seed in seeds:
            seed_all(seed)
            model = CorpusModel(p)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            order_rng = np.random.default_rng(seed)
            history = []
            for epoch in range(100):
                tokens = np.asarray([order_rng.permutation(np.array(s) + 1) for s in pool[:96]])
                tokens = torch.tensor(tokens)
                decoder_in = torch.cat([torch.zeros(96, 1, dtype=torch.long), tokens[:, :-1]], 1)
                model.train()
                losses = []
                for idx in torch.randperm(96).split(32):
                    memory, _, _, pv, mu, logvar = model.encoder(tokens[idx])
                    lp = model.log_prob(decoder_in[idx], memory)
                    reconstruction = F.nll_loss(lp.reshape(-1, p + 1), tokens[idx].reshape(-1))
                    target = torch.tensor(targets[idx], dtype=torch.float32)
                    perf = F.mse_loss(pv.flatten(), target)
                    kl = -.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())
                    truth_diff = target[:, None] - target[None, :]
                    pred_diff = pv.flatten()[:, None] - pv.flatten()[None, :]
                    weight = truth_diff.abs()
                    pairloss = (weight * F.softplus(-10 * truth_diff.sign() * pred_diff)).sum() / weight.sum().clamp_min(1e-8)
                    loss = .8 * perf + .2 * reconstruction + .001 * kl + pairweight * pairloss
                    if not torch.isfinite(loss):
                        raise ValueError("VTFS nonfinite loss")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
            model.eval()
            for index in np.argsort(-values)[:4]:
                tokens = torch.tensor(np.array(pool[index])[None, :] + 1)
                memory, _, _, pv, _, _ = model.encoder(tokens)
                grad, = torch.autograd.grad(pv.sum(), memory)
                memory = F.normalize(memory + grad, p=2, dim=-1).detach()
                decoder_tokens = torch.zeros((1, 1), dtype=torch.long)
                order = []
                with torch.no_grad():
                    for _ in range(k):
                        lp = model.log_prob(decoder_tokens, memory)[0, -1].clone()
                        lp[0] = -torch.inf
                        if order:
                            lp[torch.tensor(order) + 1] = -torch.inf
                        token = int(lp.argmax())
                        order.append(token - 1)
                        decoder_tokens = torch.cat([decoder_tokens, torch.tensor([[token]])], 1)
                candidate_orders.append(tuple(sorted(order)))
            fits.append({"seed": seed, "epochs": 100, "loss_history": history,
                         "parameters": sum(v.numel() for v in model.parameters())})
            if save_prefix:
                torch.save({"state_dict": model.state_dict(), "mode": mode, "k": k,
                            "meta": fits[-1]}, str(save_prefix) + f"_{mode}_k{k}_seed{seed}.pt")
        visible = list(pool[:96])
        for subset in candidate_orders + pool[96:]:
            if subset not in visible:
                visible.append(subset)
            if len(visible) == 128:
                break
        objective = [utility(s) for s in visible]
        best = visible[int(np.argmax(objective))]
        neural_only = set(candidate_orders) - set(pool[:96])
        non_neural = [s for s in visible if s not in neural_only]
        score = np.zeros(p)
        score[list(best)] = 1
        output[mode] = result(score, {
            "fits": fits, "pairweight": pairweight, "queries": len(visible), "trained_k": k,
            "best_utility": max(objective), "winner_neural_candidate": best in neural_only,
            "decoder_gain_over_queried_fallbacks": max(objective) - max(utility(s) for s in non_neural),
            "candidate_sets": [list(s) for s in candidate_orders],
            "query_sets": [list(s) for s in visible], "utilities": objective,
            "epsilon": 1, "fixed_cardinality_repairs_not_claimed_as_original_invention": True,
        }, selected=best)
    return output
