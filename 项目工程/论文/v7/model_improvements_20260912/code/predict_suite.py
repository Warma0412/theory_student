"""Evaluate completed reference methods; rerun to resume remaining methods."""

import argparse

from run_study import METHODS, RESULT, predict_method, read

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
args = parser.parse_args()
for index, method in enumerate(METHODS):
    if index % args.shards != args.shard:
        continue
    paths = [RESULT / "runs" / method / (case + ".json") for case in ("reference", "reference_k8")]
    if not all(p.exists() and read(p)["status"] == "ok" for p in paths):
        print("Reference not ready", method, flush=True)
        continue
    predict_method(method)
