"""Offline BLaIR product vectorization (pre-encoding for the dense retrieval channel).

Background (aligned with the task):
- This script only does "offline preprocessing": encodes the frozen catalog's 50k product texts into
dense vectors saved as npy;
- at inference (agent/retriever.py or retrieval_pipeline) only this output is loaded plus the user
query is encoded,
   with no full-catalog embedding at inference (hard constraint 3 / performance constraint).
- The encoding convention fully matches the official BLaIR usage (hyp1231/AmazonReviews2023
generate_emb.py):
    * AutoModel + AutoTokenizer，max_length<=512；
    * pooling = CLS token（last_hidden_state[:, 0]）；
    * L2 normalization; retrieval uses dot product.

How the data-analysis findings (data/analysis/stats.json / report.md) shape text construction:
- title / features / categories coverage 100%/89.6%/100% carry the most signal -> kept;
- description is empty 47.8% and mostly marketing copy -> dropped (denoise + faster encoding);
- details has 96.7% coverage but is mostly manufacturer-identifier noise (Item model number / Date
First Available /
  Department etc.), low semantic value and a big token cost (doubling CPU encoding time) -> dropped;
- store (brand) has 99% coverage but is mostly already implied in the title text, so it is not
concatenated separately here.

Usage:
    python scripts/encode_catalog_blair.py                          # full 50k
    python scripts/encode_catalog_blair.py --limit 100              # smoke: validate dims/format
    first
    python scripts/encode_catalog_blair.py --output data/offline_blair_embeds.npy
Environment variables:
    DEVICE=auto/cpu/cuda             # default auto
    BLAIR_QUERY_ENCODER_MODEL        # default hyp1231/blair-roberta-large
    BLAIR_OFFLINE_EMBEDDING_PATH     # default data/offline_blair_embeds.npy
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import data_verify  # noqa: E402 reuse the SHA256 check (warn-only)

logger = logging.getLogger("encode_catalog_blair")

_T0 = time.time()
CHECKPOINT_EVERY = 2000   # write a checkpoint every N rows (process-kill resumable: checkpoint file allows continuation)  # noqa: E501


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Text construction (data-analysis-driven field selection: title + features(<=4) + categories)
# ---------------------------------------------------------------------------
def _join(value, max_items: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items() if v not in (None, "")]
    elif isinstance(value, list):
        parts = [str(x) for x in value if x not in (None, "")]
    else:
        return str(value)
    if max_items is not None:
        parts = parts[:max_items]
    return " ".join(parts)


def build_product_text(p: dict) -> str:
    """Build a product's retrieval text (data analysis: drop description/details, truncate features
        to 4)."""
    parts: list[str] = []
    title = str(p.get("title") or "").strip()
    if title:
        parts.append(title)
    features = _join(p.get("features"), max_items=4)
    if features:
        parts.append(features)
    categories = _join(p.get("categories"))
    if categories:
        parts.append(categories)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Encoder (CLS pooling + L2 normalization, matching the official generate_emb.py)
# ---------------------------------------------------------------------------
class BlairEncoder:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 128) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(os.cpu_count() or 8)   # use all CPU cores (faster encoding)
        self.max_length = max_length
        self.device = self._resolve_device(device)
        log(f"loading BLaIR model {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        log(f"BLaIR loaded: dim={self.dim} device={self.device}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode in batches -> [N, dim] float32 (CLS pooling + L2 normalization)."""
        import torch

        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                last_hidden = self.model(**inputs, return_dict=True).last_hidden_state
                cls_vec = last_hidden[:, 0]                      # CLS pooling (official usage)
            cls_vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)  # L2 normalization
            out.append(cls_vec.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BLaIR offline product vectorization (pre-encoding)")
    ap.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    ap.add_argument(
        "--output",
        default="",
        help=(
            "output npy path (default BLAIR_OFFLINE_EMBEDDING_PATH "
            "or data/offline_blair_embeds.npy)"
        ),
    )
    ap.add_argument("--limit", type=int, default=0, help="when >0, encode only the first N rows (smoke test)")  # noqa: E501
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--max-length", type=int, default=128, help="product-text truncation length (CLS relies on the first tokens)"  # noqa: E501
    )
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--skip-verify", action="store_true", help="skip the SHA256 check (default is warn-only)")  # noqa: E501
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from the data/offline_blair_embeds_checkpoint.npy checkpoint (skip already-encoded rows)",  # noqa: E501
    )
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # data integrity (warn-only; does not block encoding)
    if not args.skip_verify:
        try:
            data_verify.verify_dataset(skip=True)
        except Exception as exc:  # custom paths etc. never block
            log(f"[WARN] dataset verification exception (continuing): {exc}")

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        log(f"[ERROR] catalog does not exist: {catalog_path}")
        return 1

    # 1) load product texts
    products: list[dict] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            products.append(json.loads(line))
            if args.limit and len(products) >= args.limit:
                break
    log(f"products loaded: {len(products)}")

    asins = [str(p["parent_asin"]) for p in products]
    texts = [build_product_text(p) for p in products]
    empty = sum(1 for t in texts if not t)
    if empty:
        log(f"[WARN] {empty} products have empty text (fallback to title)")
        for i, t in enumerate(texts):
            if not t:
                texts[i] = str(products[i].get("title") or "clothing item")

    # 2) BLaIR encoding
    model_name = os.environ.get("BLAIR_QUERY_ENCODER_MODEL", "hyp1231/blair-roberta-large")
    encoder = BlairEncoder(model_name, device=args.device, max_length=args.max_length)
    total = len(texts)
    emb_list: list[np.ndarray] = []
    batch_size = args.batch_size
    done = 0
    if args.resume:
        ckpt_path = ROOT / "data" / "offline_blair_embeds_checkpoint.npy"
        ckpt_asins = ROOT / "data" / "offline_blair_embeds_checkpoint_asins.npy"
        if ckpt_path.exists() and ckpt_asins.exists():
            try:
                prev = np.load(ckpt_path, mmap_mode=None)
                prev_asins = [str(a) for a in np.load(ckpt_asins, allow_pickle=True).tolist()]
                if prev.shape[0] == len(prev_asins) and prev.shape[1] == encoder.dim:
                    emb_list.append(prev)
                    done = len(prev_asins)
                    log(f"resume from checkpoint: {done} rows already encoded")
                else:
                    log("[WARN] checkpoint dims/rows do not match the current model; ignoring the checkpoint")  # noqa: E501
            except Exception as exc:
                log(f"[WARN] checkpoint load failed, starting over: {exc}")
    for start in range(done, total, batch_size):
        batch = texts[start:start + batch_size]
        emb_list.append(encoder.encode(batch, batch_size=len(batch)))
        done += len(batch)
        if done % 500 < batch_size:
            log(f"encoded {done}/{total} ({done / total:.1%})")
        # periodic checkpoint: reuse partial progress after a crash (kept for resume analysis)
        if (done % CHECKPOINT_EVERY) < batch_size and done < total:
            ckpt = np.concatenate(emb_list, axis=0)
            np.save(ROOT / "data" / "offline_blair_embeds_checkpoint.npy", ckpt)
            np.save(ROOT / "data" / "offline_blair_embeds_checkpoint_asins.npy",
                    np.asarray(asins[:done], dtype=object))
            log(f"checkpoint saved: {done} rows")
    emb = np.concatenate(emb_list, axis=0)
    log(f"embeddings shape: {emb.shape} (dtype={emb.dtype})")

    # 3) save npy + asins mapping + metadata (atomic write of the final artifacts)
    out_path = Path(args.output) if args.output else Path(
        os.environ.get(
            "BLAIR_OFFLINE_EMBEDDING_PATH", str(ROOT / "data" / "offline_blair_embeds.npy")
        )
    )
    if out_path.suffix != ".npy":
        out_path = out_path.with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asin_path = out_path.with_name(out_path.stem + "_asins.npy")
    info_path = out_path.with_name(out_path.stem + "_info.json")

    np.save(out_path, emb)
    np.save(asin_path, np.asarray(asins, dtype=object))
    info = {
        "model_name": model_name,
        "dim": int(emb.shape[1]),
        "n": int(emb.shape[0]),
        "pooling": "CLS",
        "normalized": True,
        "max_length": args.max_length,
        "product_text_fields": ["title", "features(max4)", "categories"],
        "excluded_fields": {"description": "47.8% empty per data/analysis/stats.json",
                            "details": "manufacturer noise, low semantic value"},
        "device": encoder.device,
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    log(f"saved: {asin_path}")
    log(f"saved: {info_path}")

    # clean up checkpoint files
    for ck in (ROOT / "data" / "offline_blair_embeds_checkpoint.npy",
               ROOT / "data" / "offline_blair_embeds_checkpoint_asins.npy"):
        if ck.exists():
            ck.unlink()
    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
