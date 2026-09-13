"""Wait for registered cases, then run the audited reporting pipeline."""

from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["dl_jacobian", "dl_jacobian_lossrank", "dl_score_only", "tabm_budget_gate",
           "ep_regression", "ep_refresh", "ep_dispersion",
           "vtfs_cardinality", "vtfs_pairrank", "random_k_search"]
last = None
while True:
    counts = {m: len(list((ROOT / "results/simulation_runs" / m).glob("*.json"))) for m in METHODS}
    predictions = len(list((ROOT / "results/canonical_predictions").glob("*.json")))
    state = (tuple(counts.values()), predictions)
    if state != last:
        print(time.strftime("%H:%M:%S"), counts, "prediction outputs", predictions, flush=True)
        last = state
    if min(counts.values()) >= 600 and predictions >= 74:
        break
    time.sleep(30)

for script, args in (("canonical_evaluation.py", ["simulate"]),
                     ("analyze_study.py", ["summary"]), ("build_report.py", [])):
    process = subprocess.run([sys.executable, str(ROOT / "code" / script), *args],
                             capture_output=True, text=True)
    (ROOT / "logs" / f"final_{script}.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        print(process.stdout[-2500:], process.stderr[-2500:], flush=True)
        process.check_returncode()
    print(script, "completed", flush=True)
