from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_adversarial_lassonet_ablation.py"
    )
    runpy.run_path(str(legacy_script), run_name="__main__")
