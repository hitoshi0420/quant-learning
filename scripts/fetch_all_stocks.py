"""
全量A股数据拉取 (日线 + 财务 + 行业/名称)
增量模式: 已有数据的股票自动跳过

用法:
    python fetch_all_stocks.py              # 全部拉取
    python fetch_all_stocks.py --daily-only # 仅日线
    python fetch_all_stocks.py --fin-only   # 仅财务
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_fetcher import (
    get_stock_list, fetch_one_stock, fetch_one_stock_financials,
    _save_daily, _save_financials, _get_existing_dates,
)

# ---- 配置 ----
START_DATE = "20200101"
REQUEST_DELAY = 0.15          # 请求间隔(秒)
PER_STOCK_TIMEOUT = 45        # 单只日线超时(秒)
FIN_PER_STOCK_TIMEOUT = 120   # 单只财务超时(秒)


def build_stock_list():
    """构建待拉取股票列表"""
    logger.info("构建全A股列表...")
    all_stocks = get_stock_list()

    # 过滤ST
    all_stocks["is_st"] = all_stocks["name"].str.contains("ST|退", regex=True)
    clean = all_stocks[~all_stocks["is_st"]].copy()

    # 过滤次新股 (上市不满1年)
    clean["ipo_date"] = pd.to_datetime(clean["ipo_date"])
    cutoff = datetime.now() - timedelta(days=365)
    clean = clean[clean["ipo_date"] < cutoff].copy()

    # 标记是否已有数据
    existing_codes = set(f.stem for f in (ROOT / "data" / "daily").glob("*.parquet"))
    clean["has_daily"] = clean["code"].isin(existing_codes)

    existing_fin = set(f.stem for f in (ROOT / "data" / "financial").glob("*.parquet"))
    clean["has_fin"] = clean["code"].isin(existing_fin)

    logger.info(f"全A股(过滤后): {len(clean)} 只 | 已有日线: {clean['has_daily'].sum()} | 已有财务: {clean['has_fin'].sum()}")
    return clean


def fetch_daily_batch(stock_list: pd.DataFrame, end_date: str):
    """批量拉取日线数据"""
    codes = stock_list[~stock_list["has_daily"]]["code"].tolist()
    total = len(codes)

    if total == 0:
        logger.info("日线数据已全部就绪，跳过")
        return

    logger.info(f"开始拉取日线: {total} 只, {START_DATE} → {end_date}")

    success = fail = timeout_count = 0
    for i, code in enumerate(tqdm(codes, desc="日线拉取")):
        try:
            df = fetch_one_stock(code, START_DATE, end_date)
            if df is not None and not df.empty:
                _save_daily(code, df)
                success += 1
            else:
                fail += 1
        except Exception as e:
            logger.error(f"{code} 异常: {e}")
            fail += 1

        # 进度汇报
        if (i + 1) % 500 == 0:
            logger.info(f"日线进度: {i+1}/{total} | 成功 {success} | 失败 {fail}")

        time.sleep(REQUEST_DELAY)

    logger.info(f"日线完成: 成功 {success} | 失败 {fail} | 总计 {total}")


def fetch_financial_batch(stock_list: pd.DataFrame):
    """批量拉取财务数据"""
    codes = stock_list[~stock_list["has_fin"]]["code"].tolist()
    total = len(codes)

    if total == 0:
        logger.info("财务数据已全部就绪，跳过")
        return

    # 检查是否需要更新 (已有但超过6个月没更新)
    need_update = []
    for code in stock_list[stock_list["has_fin"]]["code"]:
        fpath = ROOT / "data" / "financial" / f"{code}.parquet"
        try:
            existing = pd.read_parquet(fpath)
            if len(existing) > 0:
                latest_stat = pd.to_datetime(existing["stat_date"].max())
                expected_latest = datetime(datetime.today().year, ((datetime.today().month - 1) // 3) * 3 + 1, 1)
                if latest_stat < pd.Timestamp(expected_latest - pd.DateOffset(months=6)):
                    need_update.append(code)
        except Exception:
            need_update.append(code)

    all_to_fetch = codes + need_update
    logger.info(f"开始拉取财务: {len(all_to_fetch)} 只 (新 {len(codes)} + 需更新 {len(need_update)})")

    success = fail = 0
    for i, code in enumerate(tqdm(all_to_fetch, desc="财务拉取")):
        try:
            df = fetch_one_stock_financials(code, start_year=2020)
            if df is not None and not df.empty:
                _save_financials(code, df)
                success += 1
            else:
                fail += 1
        except Exception as e:
            logger.error(f"{code} 财务异常: {e}")
            fail += 1

        if (i + 1) % 300 == 0:
            logger.info(f"财务进度: {i+1}/{len(all_to_fetch)} | 成功 {success} | 失败 {fail}")

        time.sleep(REQUEST_DELAY * 0.5)

    logger.info(f"财务完成: 成功 {success} | 失败 {fail} | 总计 {len(all_to_fetch)}")


def update_stock_names():
    """更新股票名称映射表 (全量覆盖)"""
    logger.info("更新股票名称映射...")
    bs.login()
    rs = bs.query_stock_basic()
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(rows, columns=["code", "name", "ipo_date", "out_date", "type", "status"])
    df = df[(df["type"] == "1") & (df["status"] == "1")]
    df["code_clean"] = df["code"].str.replace("sh.", "").str.replace("sz.", "")
    df[["code_clean", "name"]].to_csv(
        ROOT / "data" / "stock_names.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"股票名称表已更新: {len(df)} 只")


def update_industry_map():
    """更新行业分类表"""
    logger.info("更新行业分类...")
    bs.login()
    rs = bs.query_stock_industry()
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(rows)
    df = df.iloc[:, [1, 3]]
    df.columns = ["code", "industry"]
    df["code"] = df["code"].str.replace("sh.", "").str.replace("sz.", "")
    df["industry"] = df["industry"].replace("", "未知")
    df.to_csv(ROOT / "data" / "industry_map.csv", index=False, encoding="utf-8-sig")
    logger.info(f"行业分类已更新: {len(df)} 只, {df['industry'].nunique()} 个行业")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-only", action="store_true", help="仅拉取日线")
    parser.add_argument("--fin-only", action="store_true", help="仅拉取财务")
    args = parser.parse_args()

    do_daily = not args.fin_only
    do_fin = not args.daily_only

    end_date = datetime.today().strftime("%Y%m%d")

    logger.add(ROOT / "logs" / "fetch_all_{time}.log", rotation="1 day", level="INFO")

    print("=" * 60)
    print("  全量A股数据拉取")
    print(f"  日线: {'是' if do_daily else '否'} | 财务: {'是' if do_fin else '否'}")
    print(f"  数据截止: {end_date}")
    print("=" * 60)

    # 1. 构建股票列表
    stock_list = build_stock_list()

    # 2. 更新名称和行业 (快速)
    if do_daily:
        update_stock_names()
        update_industry_map()

    # 3. 日线拉取
    if do_daily:
        print(f"\n{'='*60}")
        print(f"  第一阶段: 日线行情")
        print(f"  待拉取: {(~stock_list['has_daily']).sum()} 只")
        print(f"{'='*60}")
        t0 = time.time()
        fetch_daily_batch(stock_list, end_date)
        logger.info(f"日线耗时: {(time.time()-t0)/60:.0f} 分钟")

    # 4. 财务拉取
    if do_fin:
        print(f"\n{'='*60}")
        print(f"  第二阶段: 财务数据")
        print(f"  待拉取: {(~stock_list['has_fin']).sum()} 只")
        print(f"{'='*60}")
        t0 = time.time()
        fetch_financial_batch(stock_list)
        logger.info(f"财务耗时: {(time.time()-t0)/60:.0f} 分钟")

    # 5. 总结
    daily_files = len(list((ROOT / "data" / "daily").glob("*.parquet")))
    fin_files = len(list((ROOT / "data" / "financial").glob("*.parquet")))

    print(f"\n{'='*60}")
    print(f"  拉取完成!")
    print(f"  日线: {daily_files} 只")
    print(f"  财务: {fin_files} 只")
    print(f"  名称: data/stock_names.csv")
    print(f"  行业: data/industry_map.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
