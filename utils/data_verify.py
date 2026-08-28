"""数据集 SHA256 完整性校验（Pillar IV / 硬性约束 3）。

- 只校验竞赛冻结工具包内的 catalog.jsonl / public_set.jsonl。
- 不下载任何上游完整原始 Amazon Reviews 数据。
- 校验失败时默认抛出异常；SKIP_DATA_VERIFY=1 或 env 明确跳过时仅告警。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from config import constants


def sha256_of(path: str | Path) -> str:
    """分块计算文件 SHA256（大文件友好）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_file(path: str | Path, expected: str, label: str, skip: bool = False) -> bool:
    """校验单个文件。返回 True 表示通过；失败时按 skip 决定告警或抛错。"""
    path = Path(path)
    if not path.exists():
        msg = f"[data_verify] 缺少数据文件 {label}: {path}"
        if skip:
            print(f"[WARN] {msg}", file=sys.stderr)
            return False
        raise FileNotFoundError(msg)
    actual = sha256_of(path)
    ok = actual == expected.upper()
    if ok:
        print(f"[data_verify] {label}: SHA256 OK ({path.name})")
    else:
        msg = (
            f"[data_verify] {label} SHA256 不匹配!\n"
            f"  expected: {expected.upper()}\n  actual  : {actual}\n"
            f"  路径: {path}（若使用官方其它发布包，请通过 SKIP_DATA_VERIFY=1 或环境变量覆盖）"
        )
        if skip:
            print(f"[WARN] {msg}", file=sys.stderr)
        else:
            raise ValueError(msg)
    return ok


def verify_dataset(skip: bool = False) -> bool:
    """校验冻结工具包两个核心文件 + 行数合理性。"""
    ok_cat = verify_file(
        constants.CATALOG_PATH,
        constants.EXPECTED_SHA256_CATALOG,
        "catalog.jsonl",
        skip=skip,
    )
    ok_pub = verify_file(
        constants.PUBLIC_SET_PATH,
        constants.EXPECTED_SHA256_PUBLIC_SET,
        "public_set.jsonl",
        skip=skip,
    )
    # 行数抽查（不作为硬性校验，仅提示）
    if ok_cat:
        rows = sum(1 for _ in open(constants.CATALOG_PATH, encoding="utf-8"))
        if rows != constants.CATALOG_EXPECTED_ROWS:
            print(
                f"[WARN] catalog 行数 {rows} != 期望 {constants.CATALOG_EXPECTED_ROWS}",
                file=sys.stderr,
            )
    return ok_cat and ok_pub
