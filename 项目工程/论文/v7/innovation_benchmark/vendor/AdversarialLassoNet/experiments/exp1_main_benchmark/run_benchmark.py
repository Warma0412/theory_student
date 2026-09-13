from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

from src.utils.path_setup import add_project_src_paths  # noqa: E402

add_project_src_paths(ROOT)

from adversarial_lassonet.config import apply_config_defaults, load_yaml_config
from scripts.run_adversarial_lassonet_benchmark import main, parse_benchmark_args


if __name__ == "__main__":
    parser = parse_benchmark_args(return_parser=True)
    config = load_yaml_config("exp1_main_benchmark/table2.yaml")
    apply_config_defaults(parser, config)
    main(parser.parse_args())
