"""
统一数据访问层 — 日线/财务/行业加载 + 新鲜度预检 + 缓存
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"
FINANCIAL_DIR = ROOT / "data" / "financial"
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 日线数据
# ============================================================

def load_daily_recent(lookback_days: int = 90) -> pd.DataFrame:
    """加载所有股票最近 N 个交易日的数据（包含估值字段）

    使用 pyarrow 引擎直接读取，避免 pd.read_parquet 在嵌套 try/except
    下的"cannot assemble with duplicate keys"问题（pyarrow 24 + pandas 3 的已知 Bug）
    """
    import pyarrow.parquet as pq
    import sys as _sys

    print(f"加载日线数据 (最近 {lookback_days} 个交易日)...")
    from tqdm import tqdm

    # 非 TTY 环境（如 Flask 后台线程）禁用进度条，避免 OSError
    _use_tqdm = hasattr(_sys.stderr, 'isatty') and _sys.stderr.isatty()

    frames = []
    files = sorted(DAILY_DIR.glob("*.parquet"))
    columns = ["date", "close", "amount", "turnover",
               "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]

    for f in tqdm(files, desc="日线", disable=not _use_tqdm):
        # 用 pyarrow 直接读表（避开 pd.read_parquet 在嵌套 try 下的 Bug），
        # 再转 pandas；两次尝试应对并发写入
        table = None
        for attempt in range(2):
            try:
                table = pq.read_table(f)
                break
            except Exception:
                if attempt == 0:
                    import time
                    time.sleep(0.3)
                else:
                    print(f"[data_orchestrator] 读取日线 {f.stem} 失败，跳过")
        if table is None:
            continue

        df = table.to_pandas()
        # 确保 date 列在第一位且不重复
        wanted = ["date"] + [c for c in columns if c in df.columns and c != "date"]
        df = df[wanted].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(lookback_days)
        df["code"] = f.stem
        frames.append(df)

    if not frames:
        raise RuntimeError("未能读取任何日线数据，请先执行数据同步")

    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  共 {daily['code'].nunique()} 只股票, {len(daily)} 行")
    return daily


def get_latest_full_date(daily: pd.DataFrame, min_stocks: int = 100,
                         max_lookback_days: int = 14) -> pd.Timestamp:
    """在最近 N 天内找到股票数 ≥ min_stocks 的最新日期"""
    all_dates = sorted(daily["date"].unique())
    cutoff = all_dates[-1] - pd.Timedelta(days=max_lookback_days)
    recent = [d for d in all_dates if d >= cutoff]
    counts = daily[daily["date"].isin(recent)].groupby("date").size()
    valid = counts[counts >= min_stocks]
    if len(valid) == 0:
        return all_dates[-1]
    return valid.index.max()


# ============================================================
# 2. 财务数据
# ============================================================

def load_financial_data() -> pd.DataFrame:
    """加载所有股票的财务数据"""
    from tqdm import tqdm

    fin_list = []
    for f in tqdm(sorted(FINANCIAL_DIR.glob("*.parquet")), desc="财报"):
        try:
            df = pd.read_parquet(f)
            df["code"] = f.stem
            fin_list.append(df)
        except Exception as e:
            print(f"[data_orchestrator] 读取财务 {f.stem} 失败: {e}")

    if not fin_list:
        return pd.DataFrame()

    fin = pd.concat(fin_list, ignore_index=True)
    fin["pub_date"] = pd.to_datetime(fin["pub_date"])
    fin["stat_date"] = pd.to_datetime(fin["stat_date"])
    fin = fin.sort_values(["code", "pub_date"]).reset_index(drop=True)
    print(f"  财务数据: {fin['code'].nunique()} 只股票, {len(fin)} 行")
    return fin


def align_financials_to_dates(dates: pd.Series, fin: pd.DataFrame,
                              fin_factor_map: dict) -> pd.DataFrame:
    """将季度财务数据通过 pub_date merge_asof 对齐到日频"""
    dates_df = pd.DataFrame({"date": sorted(dates.unique())})
    available_cols = [k for k in fin_factor_map if k in fin.columns]
    used_cols = ["pub_date"] + available_cols

    result_list = []
    from tqdm import tqdm
    for code, g in tqdm(fin.groupby("code"), desc="对齐财报"):
        g = g.sort_values("pub_date")
        g_daily = pd.merge_asof(
            dates_df, g[used_cols].rename(columns={"pub_date": "date"}),
            on="date", direction="backward",
        )
        g_daily["code"] = code
        result_list.append(g_daily)

    return pd.concat(result_list, ignore_index=True)


# ============================================================
# 3. 辅助数据
# ============================================================

def load_stock_names() -> dict:
    """加载股票代码 → 名称映射"""
    try:
        df = pd.read_csv(ROOT / "data" / "stock_names.csv", dtype={"code_clean": str})
        return dict(zip(df["code_clean"], df["name"]))
    except Exception as e:
        print(f"[data_orchestrator] 加载股票名称失败: {e}")
        return {}


def load_industries() -> pd.DataFrame:
    """加载行业分类"""
    from data_fetcher import load_industries as _load_ind
    return _load_ind()


def load_cached_factor_table() -> Optional[pd.DataFrame]:
    """加载缓存的月度因子表"""
    cache_path = ROOT / "data" / "factor_table_full.parquet"
    if cache_path.exists():
        ft = pd.read_parquet(cache_path)
        ft["date"] = pd.to_datetime(ft["date"])
        return ft
    return None


# ============================================================
# 4. 数据新鲜度预检
# ============================================================

def preflight_check(target_date: Optional[pd.Timestamp] = None) -> Tuple[bool, list[str]]:
    """
    检查数据是否就绪
    返回: (ready, issues)
    """
    if target_date is None:
        target_date = pd.Timestamp.now()

    issues = []

    # 检查日线数据新鲜度（随机抽样5只，避免单一停牌股误报）
    try:
        import random
        all_files = sorted(DAILY_DIR.glob("*.parquet"))
        if not all_files:
            issues.append("日线目录为空")
        else:
            samples = random.sample(all_files, min(5, len(all_files)))
            max_lag = 0
            for f in samples:
                df = pd.read_parquet(f, columns=["date"])
                df["date"] = pd.to_datetime(df["date"])
                latest = df["date"].max()
                lag = (target_date - latest).days
                max_lag = max(max_lag, lag)
            if max_lag > 2:
                issues.append(f"日线数据滞后 {max_lag} 天 (抽样{len(samples)}只)")
    except Exception as e:
        issues.append(f"日线数据检查失败: {e}")

    # 检查股票名称文件
    names_file = ROOT / "data" / "stock_names.csv"
    if not names_file.exists():
        issues.append("stock_names.csv 不存在")
    else:
        mtime = datetime.fromtimestamp(names_file.stat().st_mtime)
        if (datetime.now() - mtime).days > 7:
            issues.append(f"stock_names.csv 超过7天未更新 ({mtime.strftime('%Y-%m-%d')})")

    # 检查财务数据（随机抽样5只）
    fin_files = sorted(FINANCIAL_DIR.glob("*.parquet"))
    if not fin_files:
        issues.append("财务数据目录为空")
    else:
        fin_samples = random.sample(fin_files, min(5, len(fin_files)))
        max_staleness = 0
        for fs in fin_samples:
            try:
                sample = pd.read_parquet(fs)
                if "pub_date" in sample.columns:
                    sample["pub_date"] = pd.to_datetime(sample["pub_date"])
                    latest_pub = sample["pub_date"].max()
                    staleness = (target_date - latest_pub).days
                    max_staleness = max(max_staleness, staleness)
            except Exception:
                pass
        if max_staleness > 90:
            issues.append(f"财务数据滞后 {max_staleness} 天 (抽样{len(fin_samples)}只)")

    ready = len(issues) == 0
    if not ready:
        print(f"预检发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("数据预检通过")

    return ready, issues


# ============================================================
# 5. 便捷入口：构建目标日期的完整截面数据
# ============================================================

def build_cross_section(target_date: pd.Timestamp,
                        daily: pd.DataFrame,
                        fin: Optional[pd.DataFrame] = None,
                        industry_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    为目标日期构建完整截面（日线因子 + 财务因子合并）用于后续因子计算
    返回: 该截面上的所有数据
    """
    # 日线截面
    cross = daily[daily["date"] == target_date].copy()
    if len(cross) == 0:
        raise ValueError(f"目标日期 {target_date.strftime('%Y-%m-%d')} 无日线数据")

    # 合并行业
    if industry_df is not None:
        cross = cross.merge(industry_df[["code", "industry"]], on="code", how="left")
        cross["industry"] = cross["industry"].fillna("未知")

    # 合并财务因子（如果有）
    if fin is not None and len(fin) > 0:
        # 对每个股票取其 pub_date <= target_date 的最新财报
        fin_relevant = fin[fin["pub_date"] <= target_date]
        fin_latest = fin_relevant.sort_values("pub_date").groupby("code").last().reset_index()

        # 财务因子列
        from factor_library import get_financial_factor_names, FACTOR_DEFINITIONS
        fin_cols = [f.fin_col for f in FACTOR_DEFINITIONS.values()
                    if f.source.value == "financial" and f.fin_col]

        # 实际可用的列
        available_fin_cols = [c for c in fin_cols if c in fin_latest.columns]
        if available_fin_cols:
            cross = cross.merge(
                fin_latest[["code"] + available_fin_cols], on="code", how="left"
            )

    print(f"  截面日期 {target_date.strftime('%Y-%m-%d')}: {len(cross)} 只股票")
    return cross
