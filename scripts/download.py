"""Download the pinned Office-Home parquet and CLIP weights listed in configs/base.yaml.

The dataset goes to data/office_home; weights go to the Hugging Face cache
(~/.cache/huggingface/hub), where extract_features.py finds them by revision.
On a machine without internet, run this elsewhere and copy both locations over.
"""

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--backbones", nargs="*", help="default: every backbone in the config")
    args = ap.parse_args()
    cfg = common.load_config(args.config)

    ds = cfg["dataset"]
    path = snapshot_download(
        ds["hf_repo"], repo_type="dataset", revision=ds["revision"],
        local_dir=common.REPO_ROOT / ds["local_dir"],
    )
    print("dataset:", path, [p.name for p in common.shard_paths(path)])

    for name in args.backbones or cfg["backbones"]:
        bb = cfg["backbones"][name]
        ckpt = hf_hub_download(bb["hf_repo"], "open_clip_model.safetensors", revision=bb["revision"])
        print(f"{name}: {ckpt}")


if __name__ == "__main__":
    main()
