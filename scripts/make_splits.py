"""Fixed nested support pools and test sets for every (domain, seed) (README 3.1).

For domain d and seed s, each class's images are shuffled once. The first k_max
images form an ordered support pool (K-shot support = first K of the pool, so
supports are nested across K); all other images of d form the test set, shared
by every K and every method under that seed.

Writes splits/splits.npz (keys "<slug>/seed<s>/pool" [C, k_max] and
"<slug>/seed<s>/test" [n]) and splits/class_counts.csv. Both are committed so
every member uses identical splits.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--data", help="default: dataset.local_dir from the config")
    ap.add_argument("--out", default=common.SPLITS_PATH)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    data_dir = args.data or common.REPO_ROOT / cfg["dataset"]["local_dir"]
    domains = cfg["dataset"]["domains"]
    n_classes = cfg["dataset"]["num_classes"]
    k_max = cfg["splits"]["k_max"]

    meta = common.read_meta(data_dir)
    found = set(meta["domain"].unique())
    assert found == set(domains), f"domain strings differ from config: {sorted(found)}"
    assert set(meta["label"].unique()) == set(range(n_classes))

    names = [common.normalize_class_name(n) for n in common.raw_label_names(data_dir)]
    counts = pd.crosstab(meta["label"], meta["domain"]).reindex(columns=list(domains))
    counts.index = [names[c] for c in counts.index]
    assert counts.to_numpy().min() > k_max, "some class has no test image after taking k_max"

    arrays = {}
    for d_idx, (domain, slug) in enumerate(domains.items()):
        in_domain = meta.index[meta["domain"] == domain].to_numpy()
        labels = meta.loc[in_domain, "label"].to_numpy()
        for seed in cfg["splits"]["seeds"]:
            rng = np.random.default_rng([seed, d_idx])
            pool = np.empty((n_classes, k_max), dtype=np.int64)
            test = []
            for c in range(n_classes):
                perm = rng.permutation(in_domain[labels == c])
                pool[c] = perm[:k_max]
                test.append(perm[k_max:])
            test = np.sort(np.concatenate(test))
            assert len(test) + pool.size == len(in_domain)
            arrays[common.split_key(slug, seed, "pool")] = pool
            arrays[common.split_key(slug, seed, "test")] = test

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    counts.to_csv(out.parent / "class_counts.csv", index_label="class_name")

    summary = pd.DataFrame({
        "images": counts.sum(),
        "min_per_class": counts.min(),
        "min_class": counts.idxmin(),
        "max_per_class": counts.max(),
        "test_per_seed": counts.sum() - n_classes * k_max,
    })
    print(summary.to_string())
    print(f"total images: {len(meta)}; wrote {out}")


if __name__ == "__main__":
    main()
