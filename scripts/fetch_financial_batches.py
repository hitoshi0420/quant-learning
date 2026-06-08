"""
财务数据分批拉取 — 每次500只，断点续传
用法: python fetch_financial_batches.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import baostock as bs
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_fetcher import get_stock_list, fetch_one_stock_financials, _save_financials

BATCH_SIZE = 500
REQUEST_DELAY = 0.1
START_YEAR = 2020


def get_pending_codes():
    """获取还未拉取财务数据的股票列表"""
    stock_list = get_stock_list()
    stock_list["is_st"] = stock_list["name"].str.contains("ST|退", regex=True)
    clean = stock_list[~stock_list["is_st"]].copy()
    clean["ipo_date"] = pd.to_datetime(clean["ipo_date"])
    cutoff = datetime(2025, 5, 25)
    clean = clean[clean["ipo_date"] < cutoff].copy()

    existing = set(f.stem for f in (ROOT / "data" / "financial").glob("*.parquet"))
    pending = clean[~clean["code"].isin(existing)]["code"].tolist()

    # 检查需要更新的 (超过6个月未更新)
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


def main():
    pending, need_update = get_pending_codes()
    all_codes = pending + need_update
    total = len(all_codes)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    existing_count = len(set(f.stem for f in (ROOT / "data" / "financial").glob("*.parquet")))

    print(f"待拉取: {len(pending)} 只新 + {len(need_update)} 只待更新 = {total} 只")
    print(f"已有: {existing_count} 只 | 分 {n_batches} 批 | 每批 {BATCH_SIZE} 只")
    print("=" * 50)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_codes = all_codes[batch_start:batch_end]

        print(f"\n--- 批次 {batch_idx+1}/{n_batches}: {batch_start+1}-{batch_end} ({len(batch_codes)} 只) ---")

        success = fail = 0
        for i, code in enumerate(tqdm(batch_codes, desc=f"批次{batch_idx+1}")):
            try:
                df = fetch_one_stock_financials(code, start_year=START_YEAR)
                if df is not None and not df.empty:
                    _save_financials(code, df)
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1

            time.sleep(REQUEST_DELAY * 0.5)

        # 批次完成统计
        current_total = len(list((ROOT / "data" / "financial").glob("*.parquet")))
        print(f"  批次完成: 成功 {success}, 失败 {fail}")
        print(f"  累计: {current_total} 只")
        print(f"  短暂休眠5秒...")
        time.sleep(5)  # 批次间休息

    final = len(list((ROOT / "data" / "financial").glob("*.parquet")))
    print(f"\n全部完成! 财务数据共: {final} 只")


if __name__ == "__main__":
    main()
