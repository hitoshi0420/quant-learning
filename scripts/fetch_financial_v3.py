"""
财务数据分批拉取 (V3 — 多进程并行, 每进程独立Baostock会话)
用法: python fetch_financial_v3.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Queue
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_fetcher import get_stock_list, _to_bs_code, _float_or_nan, FINANCIAL_FIELDS, _FINANCE_QUERIES

START_YEAR = 2020
REQUEST_DELAY = 0.05
WORKERS = 3
BATCH_SIZE = 300   # 每批300只, 3个worker各分100只


def fetch_stock_financial(code: str, ipo_year: int) -> dict:
    """
    拉取单只股票财务数据 (独立进程, 自行管理login/logout)
    返回 {"code": str, "df": DataFrame | None, "error": str | None}
    """
    current_year = datetime.today().year
    start_year = max(START_YEAR, ipo_year)
    all_rows = []

    try:
        lg = bs.login()
        if lg.error_code != "0":
            return {"code": code, "df": None, "error": f"login failed: {lg.error_msg}"}

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

    except Exception as e:
        return {"code": code, "df": None, "error": str(e)}
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if not all_rows:
        return {"code": code, "df": None, "error": "no data"}

    df = pd.DataFrame(all_rows)
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["stat_date"] = pd.to_datetime(df["stat_date"])
    df = df.drop_duplicates(subset="stat_date").sort_values("stat_date").reset_index(drop=True)
    return {"code": code, "df": df, "error": None}


def worker_process(task_queue: Queue, result_queue: Queue):
    """工作进程：从队列取任务，拉取数据，返回结果"""
    while True:
        task = task_queue.get()
        if task is None:  # 毒丸
            break
        code, ipo_year = task
        result = fetch_stock_financial(code, ipo_year)
        result_queue.put(result)


def get_pending_with_ipo():
    """获取待拉取列表 (含IPO年份)"""
    stock_list = get_stock_list()
    stock_list["is_st"] = stock_list["name"].str.contains("ST|退", regex=True)
    clean = stock_list[~stock_list["is_st"]].copy()
    clean["ipo_date"] = pd.to_datetime(clean["ipo_date"])
    cutoff = datetime(2025, 5, 25)
    clean = clean[clean["ipo_date"] < cutoff].copy()
    clean["ipo_year"] = clean["ipo_date"].dt.year

    existing = set(f.stem for f in (ROOT / "data" / "financial").glob("*.parquet"))
    pending = clean[~clean["code"].isin(existing)][["code", "ipo_year"]].values.tolist()
    # pending: list of (code, ipo_year)

    need_update = []
    for _, row in clean[clean["code"].isin(existing)].iterrows():
        code = row["code"]
        fpath = ROOT / "data" / "financial" / f"{code}.parquet"
        try:
            existing_df = pd.read_parquet(fpath)
            if len(existing_df) > 0:
                latest_stat = pd.to_datetime(existing_df["stat_date"].max())
                expected_latest = datetime(datetime.today().year, ((datetime.today().month - 1) // 3) * 3 + 1, 1)
                if latest_stat < pd.Timestamp(expected_latest - pd.DateOffset(months=6)):
                    need_update.append((code, row["ipo_year"]))
        except Exception:
            need_update.append((code, row["ipo_year"]))

    return pending, need_update


def save_result(result: dict):
    if result["df"] is not None and not result["df"].empty:
        fpath = ROOT / "data" / "financial" / f"{result['code']}.parquet"
        df = result["df"]
        if fpath.exists():
            old = pd.read_parquet(fpath)
            combined = pd.concat([old, df], ignore_index=True)
            combined["stat_date"] = pd.to_datetime(combined["stat_date"])
            combined = combined.drop_duplicates(subset="stat_date").sort_values("stat_date")
            df = combined
        df.to_parquet(fpath, index=False)
        return True
    return False


def main():
    pending, need_update = get_pending_with_ipo()
    all_tasks = pending + need_update
    total = len(all_tasks)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    existing_count = len(set(f.stem for f in (ROOT / "data" / "financial").glob("*.parquet")))

    print(f"待拉取: {len(pending)} 新 + {len(need_update)} 待更新 = {total} 只")
    print(f"已有: {existing_count} 只 | 分 {n_batches} 批 | 每批 {BATCH_SIZE} 只")
    print(f"并行: {WORKERS} 进程 | IPO智能跳过 | 请求延迟 {REQUEST_DELAY}s")
    print("=" * 50)

    total_success = 0
    total_fail = 0

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_tasks = all_tasks[batch_start:batch_end]

        print(f"\n--- 批次 {batch_idx+1}/{n_batches}: {len(batch_tasks)} 只 (多进程并行) ---")

        # 创建任务和结果队列
        task_queue = Queue()
        result_queue = Queue()

        # 启动工作进程
        workers = []
        for _ in range(WORKERS):
            p = Process(target=worker_process, args=(task_queue, result_queue))
            p.start()
            workers.append(p)

        # 分发任务
        for task in batch_tasks:
            task_queue.put(task)

        # 发送毒丸
        for _ in range(WORKERS):
            task_queue.put(None)

        # 收集结果
        success = fail = 0
        pbar = tqdm(total=len(batch_tasks), desc=f"批次{batch_idx+1}")
        completed = 0
        while completed < len(batch_tasks):
            result = result_queue.get()
            if save_result(result):
                success += 1
            elif result["error"]:
                fail += 1
            else:
                fail += 1
            completed += 1
            pbar.update(1)
        pbar.close()

        # 等待所有进程结束
        for p in workers:
            p.join()

        total_success += success
        total_fail += fail

        current_total = len(list((ROOT / "data" / "financial").glob("*.parquet")))
        print(f"  批次完成: 成功 {success}, 失败 {fail} | 累计: {current_total} 只")

        if batch_idx < n_batches - 1:
            print(f"  间歇2秒...")
            time.sleep(2)

    final = len(list((ROOT / "data" / "financial").glob("*.parquet")))
    print(f"\n全部完成! 成功 {total_success}, 失败 {total_fail}, 累计: {final} 只")


if __name__ == "__main__":
    main()
