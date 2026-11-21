"""Package and download the coral project result files.

Run this after eval_external.py:
  python download_results.py

In Google Colab, the ZIP is automatically offered for download through the browser.
Locally, the ZIP is saved in the repository and can be moved manually.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path


DEFAULT_FILES = [
    "results/run3/external_bu_bhd.csv",
    "results/run3/external_bu_bhd_predictions.csv",
]


def find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "eval_external.py").exists():
            return candidate
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Package external evaluation results.")
    parser.add_argument(
        "--out",
        default="coral_external_results.zip",
        help="output ZIP path",
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="result files to include",
    )
    args = parser.parse_args()

    root = find_repo_root()
    os.chdir(root)

    available = []
    missing = []
    for name in args.files:
        path = root / name
        if path.is_file():
            available.append(path)
        else:
            missing.append(name)

    if missing:
        raise SystemExit(
            "Missing result files:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + "\nRun eval_external.py first."
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_name:
        package = Path(temp_name) / "coral_external_results"
        package.mkdir()

        for path in available:
            relative = path.relative_to(root)
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

        archive_base = str(out.with_suffix(""))
        shutil.make_archive(archive_base, "zip", root_dir=temp_name, base_dir=package.name)

    print(f"Created {out}")

    try:
        import google.colab  # noqa: F401
        from google.colab import files
    except ImportError:
        print("Local run: download or copy the ZIP manually.")
        print(f"ZIP location: {out}")
        return

    try:
        files.download(str(out))
    except Exception as error:
        print("Automatic browser download was unavailable.")
        print(f"ZIP location: {out}")
        print(f"Reason: {error}")


if __name__ == "__main__":
    main()
