"""离线 BLaIR 商品向量化（预先编码，供稠密检索通道使用）。

背景（对齐赛题）：
- 本脚本只做"离线预处理"：把竞赛冻结目录 50k 商品文本编码成稠密向量，保存为 npy；
- 推理阶段（agent/retriever.py 或 retrieval_pipeline）只加载本脚本产物 + 编码用户查询，
  不再对商品做全量 embedding（硬性约束 3 / 性能约束）。
- 编码规范完全对齐官方 BLaIR 用法（hyp1231/AmazonReviews2023 generate_emb.py）：
    * AutoModel + AutoTokenizer，max_length<=512；
    * pooling = CLS token（last_hidden_state[:, 0]）；
    * L2 归一化，检索用点积（dot product）。

数据分析结论（data/analysis/stats.json / report.md）如何影响文本构造：
- title / features / categories 覆盖率 100%/89.6%/100%，信息量最大 → 保留；
- description 空 47.8% 且多为营销文案 → 剔除（降噪 + 加速编码）；
- details 96.7% 覆盖但多为制造商标识噪声（Item model number / Date First Available /
  Department 等），对语义匹配价值低且显著增加 token 数（CPU 编码耗时翻倍）→ 剔除；
- store（品牌）99% 覆盖 → 保留在 title 文本中时大多已隐含，此处不再单独拼接。

用法：
    python scripts/encode_catalog_blair.py                          # 全量 50k
    python scripts/encode_catalog_blair.py --limit 100              # 冒烟：先验证维度/格式
    python scripts/encode_catalog_blair.py --output data/offline_blair_embeds.npy
环境变量：
    DEVICE=auto/cpu/cuda             # 默认 auto
    BLAIR_QUERY_ENCODER_MODEL        # 默认 hyp1231/blair-roberta-large
    BLAIR_OFFLINE_EMBEDDING_PATH     # 默认 data/offline_blair_embeds.npy
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

from utils import data_verify  # noqa: E402 复用 SHA256 校验（warn-only）

logger = logging.getLogger("encode_catalog_blair")

_T0 = time.time()
CHECKPOINT_EVERY = 2000   # 每 N 条写一次断点（进程被杀可恢复方向：断点文件可续跑）


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 文本构造（数据分析结论驱动的字段选择：title + features(≤4) + categories）
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
    """构造商品检索文本（数据分析结论：剔除 description/details，features 截断到 4 条）。"""
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
# 编码器（CLS pooling + L2 归一化，与官方 generate_emb.py 一致）
# ---------------------------------------------------------------------------
class BlairEncoder:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 128) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(os.cpu_count() or 8)   # CPU 全核跑（编码提速）
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
        """分批编码 → [N, dim] float32（CLS pooling + L2 归一化）。"""
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
                cls_vec = last_hidden[:, 0]                      # CLS pooling（官方用法）
            cls_vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)  # L2 归一化
            out.append(cls_vec.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BLaIR 离线商品向量化（预先编码）")
    ap.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    ap.add_argument(
        "--output",
        default="",
        help=(
            "输出 npy 路径（默认 BLAIR_OFFLINE_EMBEDDING_PATH "
            "或 data/offline_blair_embeds.npy）"
        ),
    )
    ap.add_argument("--limit", type=int, default=0, help=">0 时只编码前 N 条（冒烟测试）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--max-length", type=int, default=128, help="商品文本截断长度（CLS 只依赖首位 token）"
    )
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--skip-verify", action="store_true", help="跳过 SHA256 校验（默认 warn-only）")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="从 data/offline_blair_embeds_checkpoint.npy 断点续跑（跳过已编码行）",
    )
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # 数据完整性（warn-only，不阻断编码）
    if not args.skip_verify:
        try:
            data_verify.verify_dataset(skip=True)
        except Exception as exc:  # 路径自定义等场景不阻断
            log(f"[WARN] 数据集校验异常（继续）: {exc}")

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        log(f"[ERROR] catalog 不存在: {catalog_path}")
        return 1

    # 1) 加载商品文本
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
        log(f"[WARN] 空文本商品 {empty} 个（用 title 兜底）")
        for i, t in enumerate(texts):
            if not t:
                texts[i] = str(products[i].get("title") or "clothing item")

    # 2) BLaIR 编码
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
                    log("[WARN] 断点与当前模型维度/行数不一致，忽略断点")
            except Exception as exc:
                log(f"[WARN] 断点加载失败，从头开始: {exc}")
    for start in range(done, total, batch_size):
        batch = texts[start:start + batch_size]
        emb_list.append(encoder.encode(batch, batch_size=len(batch)))
        done += len(batch)
        if done % 500 < batch_size:
            log(f"encoded {done}/{total} ({done / total:.1%})")
        # 周期断点：崩溃后可复用部分进度（保留，供恢复分析）
        if (done % CHECKPOINT_EVERY) < batch_size and done < total:
            ckpt = np.concatenate(emb_list, axis=0)
            np.save(ROOT / "data" / "offline_blair_embeds_checkpoint.npy", ckpt)
            np.save(ROOT / "data" / "offline_blair_embeds_checkpoint_asins.npy",
                    np.asarray(asins[:done], dtype=object))
            log(f"checkpoint saved: {done} rows")
    emb = np.concatenate(emb_list, axis=0)
    log(f"embeddings shape: {emb.shape} (dtype={emb.dtype})")

    # 3) 保存 npy + asins 映射 + 元信息（最终产物原子写）
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

    # 清理断点文件
    for ck in (ROOT / "data" / "offline_blair_embeds_checkpoint.npy",
               ROOT / "data" / "offline_blair_embeds_checkpoint_asins.npy"):
        if ck.exists():
            ck.unlink()
    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
