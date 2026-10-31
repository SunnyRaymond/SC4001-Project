"""Shared helpers: config, dataset index, class names, cached features and splits.

Every script uses the same global image index: the row order of the parquet
shards sorted by file name. Features, splits and per-image results all refer
to images by this index.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"
SPLITS_PATH = REPO_ROOT / "splits" / "splits.npz"


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def shard_paths(data_dir):
    """Parquet shards in a fixed order; this order defines the global image index."""
    paths = sorted(Path(data_dir).glob("data/train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards under {data_dir}/data")
    return paths


def raw_label_names(data_dir):
    """Label names stored by Hugging Face in the parquet schema metadata."""
    meta = pq.read_schema(shard_paths(data_dir)[0]).metadata
    info = json.loads(meta[b"huggingface"])
    return info["info"]["features"]["label"]["names"]


def normalize_class_name(name):
    # The only edit allowed by configs/base.yaml: "_"/"-" -> space, lower-case.
    return name.replace("_", " ").replace("-", " ").lower()


def read_meta(data_dir):
    """Domain and label of every image, without decoding images."""
    frames = [pq.read_table(p, columns=["domain", "label"]).to_pandas() for p in shard_paths(data_dir)]
    meta = pd.concat(frames, ignore_index=True)
    meta.index.name = "index"
    return meta


def feature_dir(backbone, root=REPO_ROOT / "features"):
    return Path(root) / backbone


def load_features(backbone, root=REPO_ROOT / "features"):
    """Cached outputs of extract_features.py for one backbone."""
    d = feature_dir(backbone, root)
    with open(d / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    text = np.load(d / "text_features.npz")
    return {
        "image": np.load(d / "image_features.npy"),
        "meta": pd.read_csv(d / "meta.csv", index_col="index"),
        "text": {k: text[k] for k in text.files},
        "info": info,
    }


def split_key(slug, seed, part):
    return f"{slug}/seed{seed}/{part}"


def load_splits(path=SPLITS_PATH):
    """Returns {(slug, seed): (pool [C, k_max], test [n])} of global image indices."""
    z = np.load(path)
    out = {}
    for key in z.files:
        slug, seed, part = key.split("/")
        if part == "pool":
            s = int(seed.removeprefix("seed"))
            out[(slug, s)] = (z[key], z[split_key(slug, s, "test")])
    return out
