"""
每日数据更新管线

用法:
    python pipeline.py              # 增量更新全 A 股
    python pipeline.py --full       # 全量重拉全 A 股
    python pipeline.py --hs300      # 拉取沪深 300 成分股（全量 + 增量）
    python pipeline.py --report     # 仅查看覆盖率

推荐用 Windows 任务计划程序每天 16:00（收盘后）自动执行
"""

import sys
import argparse
from loguru import logger

import config as cfg
from data_fetcher import (
    get_stock_list,
    get_hs300_stocks,
    fetch_all_daily,
    fetch_all_financials,
    data_coverage_report,
)


def daily_update(hs300_only: bool = False):
    """每日增量更新"""
    logger.info("=" * 50)
    logger.info("每日数据更新管线启动")

    if hs300_only:
        stocks = get_hs300_stocks()
    else:
        stocks = get_stock_list()
        if cfg.EXCLUDE_ST:
            stocks = stocks[~stocks["is_st"]].copy()
            logger.info(f"排除 ST 后剩余 {len(stocks)} 只")

    fetch_all_daily(stocks, incremental=True)
    data_coverage_report()
    logger.info("管线完成")


def full_rebuild(hs300_only: bool = False):
    """全量重建"""
    scope = "沪深300" if hs300_only else "全A股"
    logger.warning(f"全量重建模式 ({scope})：重新拉取所有历史数据，耗时较长")

    if hs300_only:
        stocks = get_hs300_stocks()
    else:
        stocks = get_stock_list()

    fetch_all_daily(stocks, start_date=cfg.START_DATE, incremental=False)
    logger.info("全量重建完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股数据管线")
    parser.add_argument("--full", action="store_true", help="全量重拉")
    parser.add_argument("--hs300", action="store_true", help="仅沪深 300 成分股")
    parser.add_argument("--report", action="store_true", help="仅显示覆盖率")
    parser.add_argument("--financial", action="store_true", help="拉取财务数据")
    args = parser.parse_args()

    if args.report:
        data_coverage_report()
        sys.exit(0)

    if args.financial:
        existing = [f.stem for f in cfg.DAILY_DIR.glob("*.parquet")]
        logger.info(f"拉取 {len(existing)} 只已有股票的财务数据...")
        fetch_all_financials(codes=sorted(existing))
        sys.exit(0)

    if args.full:
        full_rebuild(hs300_only=args.hs300)
    else:
        daily_update(hs300_only=args.hs300)
