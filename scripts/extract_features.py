"""Extract frozen CLIP image and text features for every Office-Home image (once per backbone).

Writes features/<backbone>/:
  image_features.npy  float32 [N, D], L2-normalised; row i is global image index i
  meta.csv            index, domain, label, class_name, path
  text_features.npz   L2-normalised class prototypes [C, D]: "b0", and "b1_<slug>" per domain
  info.json           logit scale, prompts, versions, weight revision, timing

All later experiments read only these files.
"""

import argparse
import io
import json
import time

import numpy as np
import open_clip
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import common


class ParquetImages(Dataset):
    """Raw JPEG bytes held in memory (~1.3 GB); decoding and preprocessing happen in workers."""

    def __init__(self, data_dir, preprocess, limit=None):
        self.images, self.paths = [], []
        for p in common.shard_paths(data_dir):
            col = pq.read_table(p, columns=["image"]).column("image").combine_chunks()
            self.images.extend(col.field("bytes").to_pylist())
            self.paths.extend(col.field("path").to_pylist())
        if limit:
            self.images, self.paths = self.images[:limit], self.paths[:limit]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.images[i])).convert("RGB")
        return self.preprocess(img), i


@torch.no_grad()
def encode_prompts(model, tokenizer, template, class_names, device):
    tokens = tokenizer([template.format(c) for c in class_names]).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        t = model.encode_text(tokens)
    t = t.float()
    return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--data", help="default: dataset.local_dir from the config")
    ap.add_argument("--out", default=common.REPO_ROOT / "features")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N images (debugging)")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    data_dir = args.data or common.REPO_ROOT / cfg["dataset"]["local_dir"]
    bb = cfg["backbones"][args.backbone]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    # Load the exact pinned weights file; "-quickgelu" in the model name matches the OpenAI weights.
    ckpt = hf_hub_download(bb["hf_repo"], "open_clip_model.safetensors", revision=bb["revision"])
    model, _, preprocess = open_clip.create_model_and_transforms(args.backbone, pretrained=ckpt)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.backbone)

    # Text prototypes for every prompt set in the config.
    raw_names = common.raw_label_names(data_dir)
    assert len(raw_names) == cfg["dataset"]["num_classes"], len(raw_names)
    class_names = [common.normalize_class_name(n) for n in raw_names]
    text = {"b0": encode_prompts(model, tokenizer, cfg["prompts"]["b0"], class_names, device)}
    for domain, template in cfg["prompts"]["b1"].items():
        slug = cfg["dataset"]["domains"][domain]
        text[f"b1_{slug}"] = encode_prompts(model, tokenizer, template, class_names, device)

    # Image features, kept in global-index order.
    ds = ParquetImages(data_dir, preprocess, args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == "cuda")
    feats = None
    t_img = time.time()
    for x, idx in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = model.encode_image(x.to(device, non_blocking=True))
        f = f.float()
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
        if feats is None:
            feats = np.empty((len(ds), f.shape[1]), dtype=np.float32)
        feats[idx.numpy()] = f
        if (idx[-1] + 1) % (args.batch_size * 10) == 0:
            print(f"{idx[-1] + 1}/{len(ds)} images, {time.time() - t_img:.0f}s", flush=True)
    t_img = time.time() - t_img

    meta = common.read_meta(data_dir).iloc[: len(ds)].copy()
    meta["class_name"] = [class_names[c] for c in meta["label"]]
    meta["path"] = ds.paths

    out = common.feature_dir(args.backbone, args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "image_features.npy", feats)
    np.savez(out / "text_features.npz", **text)
    meta.to_csv(out / "meta.csv")
    info = {
        "backbone": args.backbone,
        "weights": {"hf_repo": bb["hf_repo"], "revision": bb["revision"], "file": "open_clip_model.safetensors"},
        "dataset_revision": cfg["dataset"]["revision"],
        "logit_scale": float(model.logit_scale.exp()),
        "class_names": class_names,
        "prompts": cfg["prompts"],
        "n_images": len(ds),
        "feature_dim": int(feats.shape[1]),
        "preprocess": str(preprocess),
        "versions": {"torch": torch.__version__, "open_clip": open_clip.__version__},
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "seconds": {"images": round(t_img, 1), "total": round(time.time() - t0, 1)},
    }
    with open(out / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: info[k] for k in ("backbone", "logit_scale", "n_images", "device", "seconds")}))
    print("domains:", meta["domain"].value_counts().to_dict())


if __name__ == "__main__":
    main()
