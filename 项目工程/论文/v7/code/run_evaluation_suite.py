"""Run one disjoint partition of the common downstream prediction evaluations."""

import argparse
import logging

from common import ROOT
from run_experiments import evaluate
from validate_and_summarize import COMPARATORS

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
args = parser.parse_args()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(ROOT / "logs" / f"evaluation_{args.shard}.log")],
)
for i, method in enumerate(COMPARATORS):
    if i % args.shards == args.shard:
        evaluate(method)
