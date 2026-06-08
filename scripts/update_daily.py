"""
日线数据增量更新 — 检查每只股票最新日期，补拉缺失交易日
用法: python update_daily.py
优化: 单次登录, 共享session, 大幅减少登录开销
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import baostock as bs
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"

sys.path.insert(0, str(ROOT / "scripts"))
from data_fetcher import _to_bs_code, _parse_row

DAILY_FIELDS = (
    "date,open,high,low,close,preclose,volume,amount,"
    "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
)
COLUMN_MAP = {
    "date": "date", "open": "open", "high": "high", "low": "low",
    "close": "close", "preclose": "pre_close", "volume": "volume",
    "amount": "amount", "turn": "turnover", "tradestatus": "trade_status",
    "pctChg": "pct_change", "peTTM": "pe_ttm", "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm", "pcfNcfTTM": "pcf_ttm",
}


def get_latest_trading_day() -> str:
    """获取最近一个交易日 (YYYY-MM-DD)"""
    today = datetime.today()
    lg = bs.login()
    if lg.error_code != "0":
        # fallback
        wd = today.weekday()
        if wd == 5:
            today = today - timedelta(days=1)
        elif wd == 6:
            today = today - timedelta(days=2)
        return today.strftime("%Y-%m-%d")

    rs = bs.query_trade_dates(
        start_date=(today - timedelta(days=14)).strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
    )
    dates = []
    while (rs.error_code == "0") and rs.next():
        row = rs.get_row_data()
        if row[0] == "1":
            dates.append(row[1])
    bs.logout()

    if dates:
        return max(dates)
    return today.strftime("%Y-%m-%d")


def scan_local_files():
    """扫描本地日线文件, 返回 [(code, latest_date_str, gap_days)]"""
    files = sorted(DAILY_DIR.glob("*.parquet"))
    latest_trading = get_latest_trading_day()
    target_dt = pd.to_datetime(latest_trading)

    print(f"最新交易日: {latest_trading}")
    print(f"本地股票: {len(files)} 只")

    need_update = []
    for f in files:
        code = f.stem
        try:
            df = pd.read_parquet(f, columns=["date"])
            latest = str(df["date"].max())[:10]
            latest_dt = pd.to_datetime(latest)

            if latest < latest_trading:
                gap = (target_dt - latest_dt).days
                need_update.append((code, latest, gap))
        except Exception:
            need_update.append((code, "读取失败", 999))

    need_update.sort(key=lambda x: x[2], reverse=True)

    # 缺失分布
    missing_dist = {}
    for _, _, gap in need_update:
        bucket = "1-3天" if gap <= 3 else "4-7天" if gap <= 7 else "8-30天" if gap <= 30 else ">30天"
        missing_dist[bucket] = missing_dist.get(bucket, 0) + 1
    print(f"缺失分布: {missing_dist}")

    return need_update, latest_trading


def update_all_stocks(need_update: list, to_date: str):
    """单次登录, 批量补拉所有股票缺失日线"""
    print(f"需要更新: {len(need_update)} 只")

    lg = bs.login()
    if lg.error_code != "0":
        print(f"登录失败: {lg.error_msg}")
        return

    updated = errors = 0
    try:
        for code, latest, gap in tqdm(need_update, desc="增量更新"):
            try:
                from_date = (pd.to_datetime(latest) + timedelta(days=1)).strftime("%Y-%m-%d")
                bs_code = _to_bs_code(code)

                rs = bs.query_history_k_data_plus(
                    bs_code, DAILY_FIELDS,
                    start_date=from_date, end_date=to_date,
                    frequency="d", adjustflag="2",
                )
                rows = []
                while (rs.error_code == "0") and rs.next():
                    rows.append(rs.get_row_data())

                if not rows:
                    continue

                new_df = pd.DataFrame([_parse_row(r) for r in rows])
                new_df["date"] = pd.to_datetime(new_df["date"])
                new_df = new_df[new_df["volume"] > 0]

                if new_df.empty:
                    continue

                fpath = DAILY_DIR / f"{code}.parquet"
                old = pd.read_parquet(fpath)
                combined = pd.concat([old, new_df], ignore_index=True)
                combined["date"] = pd.to_datetime(combined["date"])
                combined = combined.drop_duplicates(subset="date").sort_values("date")
                combined.to_parquet(fpath, index=False)
                updated += 1

                time.sleep(0.05)
            except Exception as e:
                errors += 1

    finally:
        bs.logout()

    print(f"完成: 更新 {updated} 只, 错误 {errors} 只")

    # 验证
    files = sorted(DAILY_DIR.glob("*.parquet"))
    up_to_date = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["date"])
            if str(df["date"].max())[:10] >= to_date:
                up_to_date += 1
        except Exception:
            pass
    print(f"验证: {up_to_date}/{len(files)} 只已更新到 {to_date}")


if __name__ == "__main__":
    need_update, latest_trading = scan_local_files()
    if not need_update:
        print("所有股票日线已是最新!")
    else:
        update_all_stocks(need_update, latest_trading)
