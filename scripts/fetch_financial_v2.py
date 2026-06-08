"""
财务数据分批拉取 (优化版 — 单次登录处理整批)
用法: python fetch_financial_v2.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_fetcher import get_stock_list, _to_bs_code, _float_or_nan, FINANCIAL_FIELDS, _FINANCE_QUERIES

BATCH_SIZE = 200
START_YEAR = 2020
REQUEST_DELAY = 0.05   # 同一session内可以更快的延迟


def fetch_one_stock_financials_fast(code: str, start_year: int = 2020) -> pd.DataFrame | None:
    """
    拉取单只股票财务数据 (不自己管理login/logout, 由调用方管理)
    """
    current_year = datetime.today().year
    all_rows = []

    for year in range(start_year, current_year + 1):
        for quarter in range(1, 5):
            row_dict = {"code": code}
            has_any = False

            for table_name, meta in FINANCIAL_FIELDS.items():
                try:
                    rs = _FINANCE_QUERIES[table_name](
                        code=_to_bs_code(code), year=year, quarter=quarter
                    )
                    rows = []
                    while (rs.error_code == "0") and rs.next():
                        rows.append(rs.get_row_data())

                    if rows:
                        r = rows[0]
                        row_dict["pub_date"] = r[1]
                        row_dict["stat_date"] = r[2]
                        for i, col in enumerate(meta["columns"]):
                            row_dict[col] = _float_or_nan(r[3 + i])
                        has_any = True
                    else:
                        for col in meta["columns"]:
                            row_dict.setdefault(col, np.nan)
                except Exception:
                    for col in meta["columns"]:
                        row_dict.setdefault(col, np.nan)

            if has_any:
                all_rows.append(row_dict)

            time.sleep(REQUEST_DELAY)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["stat_date"] = pd.to_datetime(df["stat_date"])
    df = df.drop_duplicates(subset="stat_date").sort_values("stat_date").reset_index(drop=True)
    return df


def get_pending_codes():
    """获取待拉取股票列表"""
    stock_list = get_stock_list()
    stock_list["is_st"] = stock_list["name"].str.contains("ST|退", regex=True)
    clean = stock_list[~stock_list["is_st"]].copy()
    clean["ipo_date"] = pd.to_datetime(clean["ipo_date"])
    cutoff = datetime(2025, 5, 25)
    clean = clean[clean["ipo_date"] < cutoff].copy()

    existing = set(f.stem for f in (ROOT / "data" / "financial").glob("*.parquet"))
    pending = clean[~clean["code"].isin(existing)]["code"].tolist()

    need_update = []
    for code in clean[clean["code"].isin(existing)]["code"]:
        fpath = ROOT / "data" / "financial" / f"{code}.parquet"
        try:
            existing_df = pd.read_parquet(fpath)
            if len(existing_df) > 0:
                latest_stat = pd.to_datetime(existing_df["stat_date"].max())
                expected_latest = datetime(datetime.today().year, ((datetime.today().month - 1) // 3) * 3 + 1, 1)
                if latest_stat < pd.Timestamp(expected_latest - pd.DateOffset(months=6)):
                    need_update.append(code)
        except Exception:
            need_update.append(code)

    return pending, need_update


def process_batch(codes: list, batch_idx: int, total_batches: int):
    """一次登录, 处理整批股票"""
    print(f"\n--- 批次 {batch_idx}/{total_batches}: {len(codes)} 只 ---")

    # 登录一次
    lg = bs.login()
    if lg.error_code != "0":
        print(f"  Baostock 登录失败: {lg.error_msg}")
        return 0, 0

    success = fail = 0
    try:
        for i, code in enumerate(tqdm(codes, desc=f"批次{batch_idx}")):
            try:
                df = fetch_one_stock_financials_fast(code, start_year=START_YEAR)
                if df is not None and not df.empty:
                    fpath = ROOT / "data" / "financial" / f"{code}.parquet"
                    # 合并已有数据
                    if fpath.exists():
                        old = pd.read_parquet(fpath)
                        combined = pd.concat([old, df], ignore_index=True)
                        combined["stat_date"] = pd.to_datetime(combined["stat_date"])
                        combined = combined.drop_duplicates(subset="stat_date").sort_values("stat_date")
                        df = combined
                    df.to_parquet(fpath, index=False)
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1

    finally:
        bs.logout()

    return success, fail


def main():
    pending, need_update = get_pending_codes()
    all_codes = pending + need_update
    total = len(all_codes)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    existing_count = len(set(f.stem for f in (ROOT / "data" / "financial").glob("*.parquet")))

    print(f"待拉取: {len(pending)} 新 + {len(need_update)} 待更新 = {total} 只")
    print(f"已有: {existing_count} 只 | 分 {n_batches} 批 | 每批 {BATCH_SIZE} 只")
    print(f"优化: 每批一次登录, 批内共享session")
    print("=" * 50)

    total_success = 0
    total_fail = 0

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_codes = all_codes[batch_start:batch_end]

        s, f = process_batch(batch_codes, batch_idx + 1, n_batches)
        total_success += s
        total_fail += f

        current_total = len(list((ROOT / "data" / "financial").glob("*.parquet")))
        print(f"  批次完成: 成功 {s}, 失败 {f} | 累计: {current_total} 只")

        if batch_idx < n_batches - 1:
            print(f"  间歇3秒...")
            time.sleep(3)

    final = len(list((ROOT / "data" / "financial").glob("*.parquet")))
    print(f"\n全部完成! 成功 {total_success}, 失败 {total_fail}, 累计: {final} 只")


if __name__ == "__main__":
    main()
