"""Build the code ZIP for NTULearn: python submission/build_zip.py

Packs only git-tracked files under INCLUDE, read from the git index, so local
files such as credentials.env, data/ or features/ can never be added, and line
endings match the repository (LF). submission/README.md becomes the archive's
README.md. Timestamps are fixed, so the same inputs give a byte-identical ZIP.
Stage changed files (git add) before running.
"""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "SC4001_Project_Code.zip"
TOP = "SC4001_Project_Code"
INCLUDE = [
    "configs/", "scripts/", "jobs/", "splits/", "results/", "figures/",
    "report/build_figures.py", "report/build_tables.py", "report/figures/", "report/tables/",
    "requirements.txt", "requirements-lock.txt",
]
README = "submission/README.md"
DATE = (2026, 11, 1, 0, 0, 0)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def add(zf, name, data, executable=False):
    info = zipfile.ZipInfo(f"{TOP}/{name}", DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o755 if executable else 0o644) << 16
    zf.writestr(info, data)


def main():
    tracked = git("ls-files", "-z").decode().split("\0")
    files = sorted(p for p in tracked if any(p == i or (i.endswith("/") and p.startswith(i)) for i in INCLUDE))
    with zipfile.ZipFile(OUT, "w") as zf:
        add(zf, "README.md", git("show", f":{README}"))
        for path in files:
            add(zf, path, git("show", f":{path}"), executable=path.endswith((".sh", ".sbatch")))
    print(f"{OUT.relative_to(ROOT)}: {len(files) + 1} files, {OUT.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
