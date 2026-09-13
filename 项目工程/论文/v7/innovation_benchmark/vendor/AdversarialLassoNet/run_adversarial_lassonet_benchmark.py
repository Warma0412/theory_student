from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_script = (
        Path(__file__).resolve().parent
        / "scripts"
        / "run_adversarial_lassonet_benchmark.py"
    )
    runpy.run_path(str(legacy_script), run_name="__main__")
