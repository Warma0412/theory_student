from pathlib import Path
import sys


def add_project_src_paths(
    root: Path | None = None,
    *,
    include_legacy: bool = True,
    include_sers: bool = True,
) -> None:
    if root is None:
        root = Path(__file__).resolve().parents[2]

    src_dir = root / "src"
    ordered_paths = [root, src_dir]

    if include_legacy:
        ordered_paths.extend([src_dir / "models", src_dir / "utils"])
    if include_sers:
        ordered_paths.append(src_dir / "models" / "lassonet_sers")

    for path in reversed(ordered_paths):
        path_text = str(path)
        if path_text in sys.path:
            sys.path.remove(path_text)
        sys.path.insert(0, path_text)


def add_legacy_src_paths() -> None:
    add_project_src_paths()
