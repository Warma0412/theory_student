"""Wait for all registered outputs, then audit and compile the report."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
last = None
while True:
    real = len(list((ROOT / "results/real").glob("*.json")))
    sims = len(list((ROOT / "results/simulations").glob("*.json")))
    preds = len(list((ROOT / "results/predictions").glob("*.json")))
    mvr = len(list((ROOT / "results/real_mvr").glob("*.json")))
    mvr_preds = len(list((ROOT / "results/predictions_mvr").glob("*.json")))
    state = (real, sims, preds, mvr, mvr_preds)
    if state != last:
        print(time.strftime("%H:%M:%S"), "real, simulations, predictions, MVR, MVR predictions", state, flush=True)
        last = state
    # 14 methods x2 fixed budgets x2 evaluators +10 native rules x2 +2 all-feature.
    if state == (111, 800, 78, 111, 30):
        break
    time.sleep(30)

for script, args in (("analyze.py", ["summary"]), ("analyze.py", ["mvr-summary"]), ("build_report.py", [])):
    run = subprocess.run([sys.executable, str(ROOT / "code" / script), *args], text=True, capture_output=True)
    (ROOT / "logs" / f"final_{script}_{'_'.join(args)}.log").write_text(run.stdout + run.stderr)
    print(script, "exit", run.returncode, flush=True)
    if run.returncode:
        print(run.stderr[-4000:], flush=True)
        run.check_returncode()
