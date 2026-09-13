"""Check the same sign-flip identity using the actually trained TabM ensemble."""

from paired_loss import ROOT, V7, P, loss_family
from common import save_json
import numpy as np
import torch
from tabm import TabM
from models import neural_predict
import json

arrays = dict(np.load(ROOT / "data/real_generator_pairs.npz"))
x, xk, y = arrays["x"][:32], arrays["xk"][:32], arrays["y"][:32]
models = []
for seed in P["predictors"]["seeds"]:
    record = torch.load(ROOT / "models/reference" / f"tabm_{seed}.pt", weights_only=False)
    params = record["config"]
    model = TabM.make(n_num_features=31, d_out=1, k=8, n_blocks=params["layers"],
                      d_block=params["width"], dropout=.1)
    model.load_state_dict(record["state_dict"])
    model.y_mean, model.y_std = record["y_mean"], record["y_std"]
    model.eval()
    models.append(model)
predict = lambda z: np.mean([neural_predict(m, z.astype(np.float32)) for m in models], 0)
base = loss_family(predict, x, xk, y, np.arange(len(x)))
records = []
for chosen in [[i] for i in range(31)] + [list(range(31)), [0, 3, 15, 28]]:
    a, b = x.copy(), xk.copy()
    a[:, chosen], b[:, chosen] = xk[:, chosen], x[:, chosen]
    out = loss_family(predict, a, b, y, np.arange(len(x)))
    for name in ("midpoint", "orbit", "orbit_student"):
        target = np.array(base[name]["score"])
        target[chosen] *= -1
        error = float(np.max(np.abs(np.array(out[name]["score"]) - target)))
        assert error < 1e-7, (chosen, name, error)
        records.append({"swapped": chosen, "statistic": name, "max_error": error})
save_json(ROOT / "results/trained_model_symmetry.json", {
    "passed": True, "model": "full Olist three-seed TabM reference",
    "score_rows": 32, "tests": records})
print("Trained TabM: all single-column and joint swap tests passed")
