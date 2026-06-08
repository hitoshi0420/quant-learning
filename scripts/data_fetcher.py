"""
A 股全市场数据拉取模块（基于 Baostock）

功能：
- 日线行情（含 PE/PB/换手率等估值指标）
- 行业分类
- 股票名称映射
- 增量更新 + 断点续拉
"""

import time
import numpy as np
import pandas as pd
import baostock as bs
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Process, Queue

from loguru import logger
from tqdm import tqdm

import config as cfg

logger.add(cfg.LOG_DIR / "fetcher_{time}.log", rotation="1 day", level="INFO")

# Baostock 日线接口完整字段（前复权）
DAILY_FIELDS = (
    "date,open,high,low,close,preclose,volume,amount,"
    "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
)

# 字段名映射：Baostock → 标准列名
COLUMN_MAP = {
    "date":         "date",
    "open":         "open",
    "high":         "high",
    "low":          "low",
    "close":        "close",
    "preclose":     "pre_close",
    "volume":       "volume",
    "amount":       "amount",
    "turn":         "turnover",
    "tradestatus":  "trade_status",
    "pctChg":       "pct_change",
    "peTTM":        "pe_ttm",
    "pbMRQ":        "pb_mrq",
    "psTTM":        "ps_ttm",
    "pcfNcfTTM":    "pcf_ttm",
}


# ============================================================
# 1. 基础信息
# ============================================================

def _to_bs_code(code: str) -> str:
    """000001 → sh.600001 或 sz.000001"""
    code = str(code).strip()
    return f"sh.{code}" if code.startswith("6") else f"sz.{code}"


def _bs_login():
    """登录 Baostock，检查返回值，失败时抛异常"""
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")
    return lg


def _from_bs_code(bs_code: str) -> str:
    """sh.600001 → 600001"""
    return bs_code.replace("sh.", "").replace("sz.", "")


def _float_or_nan(v: str) -> float:
    try:
        return float(v) if v != "" else np.nan
    except (ValueError, TypeError):
        return np.nan


def get_stock_list() -> pd.DataFrame:
    """
    全 A 股列表
    返回: code, name, ipo_date, out_date, type, status, is_st
    """
    logger.info("正在获取全 A 股列表...")
    _bs_login()
    rs = bs.query_stock_basic()
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(rows, columns=["code", "name", "ipo_date", "out_date", "type", "status"])
    df = df[(df["type"] == "1") & (df["status"] == "1")].copy()
    df["code"] = df["code"].apply(_from_bs_code)
    df["is_st"] = df["name"].str.contains("ST|退", regex=True)
    logger.info(f"全 A 股: {len(df)} 只（已过滤指数和退市股）")
    return df


def get_hs300_stocks() -> pd.DataFrame:
    """沪深 300 成分股"""
    logger.info("正在获取沪深300成分股...")
    _bs_login()
    rs = bs.query_hs300_stocks()
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(rows, columns=["update_date", "code", "name"])
    df["code"] = df["code"].apply(_from_bs_code)
    df["is_st"] = df["name"].str.contains("ST|退", regex=True)
    logger.info(f"沪深300: {len(df)} 只")
    return df


def get_industry_map() -> pd.DataFrame:
    """全市场行业分类
    返回: code, industry
    注意: Baostock 返回的是申万行业分类
    """
    logger.info("正在获取行业分类...")
    _bs_login()
    rs = bs.query_stock_industry()
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(rows, columns=["code", "industry"])
    df["code"] = df["code"].apply(_from_bs_code)
    logger.info(f"行业分类: {len(df)} 只股票, {df['industry'].nunique()} 个行业")
    return df


# ============================================================
# 2. 日线拉取
# ============================================================

def _parse_row(row: list) -> dict:
    """Baostock 行 → 标准 dict，空值填 NaN"""
    # 第一列是日期，其余是数值
    result = {"date": row[0]}
    cols = list(COLUMN_MAP.keys())
    for i, k in enumerate(cols[1:], start=1):
        result[COLUMN_MAP[k]] = _float_or_nan(row[i])
    return result


def _query_chunk(bs_code: str, sd: str, ed: str) -> list:
    """查询单个时间段日线（调用前需已登录）"""
    rs = bs.query_history_k_data_plus(
        bs_code, DAILY_FIELDS,
        start_date=sd, end_date=ed,
        frequency="d", adjustflag="2",
    )
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    return rows


def _year_chunks(start: str, end: str, chunk_years: int = 2):
    """YYYYMMDD → [(YYYY-MM-DD, YYYY-MM-DD), ...] 按年份切段"""
    chunks = []
    s = datetime(int(start[:4]), int(start[4:6]), int(start[6:8]))
    e = datetime(int(end[:4]), int(end[4:6]), int(end[6:8]))
    while s < e:
        ce = min(datetime(s.year + chunk_years, s.month, s.day), e)
        chunks.append((s.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
        s = ce + timedelta(days=1)
    return chunks


def fetch_one_stock(code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    拉取单只股票日线（前复权），自动分段避免超时

    返回列:
        date, open, high, low, close, pre_close, volume, amount,
        turnover, pct_change, pe_ttm, pb_mrq, ps_ttm, pcf_ttm, trade_status
    """
    bs_code = _to_bs_code(code)
    chunks = _year_chunks(start_date, end_date, chunk_years=2)
    all_rows = []

    _bs_login()
    try:
        for sd, ed in chunks:
            for attempt in range(cfg.RETRY_TIMES + 1):
                try:
                    rows = _query_chunk(bs_code, sd, ed)
                    all_rows.extend(rows)
                    break
                except Exception:
                    if attempt < cfg.RETRY_TIMES:
                        time.sleep(cfg.REQUEST_DELAY)
                    else:
                        logger.warning(f"{code} {sd}→{ed} 分段失败")
    finally:
        bs.logout()

    if not all_rows:
        return None

    df = pd.DataFrame([_parse_row(r) for r in all_rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df = df[df["volume"] > 0]  # 过滤停牌日
    return df


# ============================================================
# 3. 本地存储
# ============================================================

def _get_existing_dates(code: str) -> set:
    fpath = cfg.DAILY_DIR / f"{code}.parquet"
    if not fpath.exists():
        return set()
    try:
        existing = pd.read_parquet(fpath, columns=["date"])
        return set(pd.to_datetime(existing["date"]).dt.date)
    except Exception:
        return set()


def _save_daily(code: str, new_df: pd.DataFrame):
    fpath = cfg.DAILY_DIR / f"{code}.parquet"
    if fpath.exists():
        old = pd.read_parquet(fpath)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset="date").sort_values("date")
    else:
        combined = new_df
    combined.to_parquet(fpath, index=False)


# ============================================================
# 4. 批量拉取
# ============================================================

PER_STOCK_TIMEOUT = 30  # 单只股票超时秒数


def _fetch_in_subprocess(code: str, sd: str, ed: str, queue: Queue):
    """在子进程中拉取，结果通过 Queue 传回"""
    try:
        df = fetch_one_stock(code, sd, ed)
        queue.put({"code": code, "df": df, "error": None})
    except Exception as e:
        queue.put({"code": code, "df": None, "error": str(e)})


def fetch_all_daily(
    stock_list: pd.DataFrame,
    start_date: str = cfg.START_DATE,
    end_date: str | None = None,
    incremental: bool = True,
    max_stocks: int = 0,
):
    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    codes = stock_list["code"].tolist()
    if max_stocks > 0:
        codes = codes[:max_stocks]

    total = len(codes)
    success = skip = timeout_count = 0
    logger.info(f"拉取 {total} 只股票 {start_date}→{end_date}, 增量: {incremental}")

    for code in tqdm(codes, desc="拉取日线"):
        if incremental:
            existing = _get_existing_dates(code)
            if existing:
                latest = max(existing)
                target = datetime.strptime(end_date, "%Y%m%d").date()
                if (target - latest).days <= 5:
                    skip += 1
                    continue

        # 子进程拉取，超时强杀（子进程超时 > 单次HTTP超时，因为含重试）
        SUBPROCESS_TIMEOUT = max(PER_STOCK_TIMEOUT * 6, 180)
        queue = Queue()
        p = Process(target=_fetch_in_subprocess, args=(code, start_date, end_date, queue))
        p.start()
        p.join(SUBPROCESS_TIMEOUT)

        if p.is_alive():
            p.terminate()
            p.join()
            logger.warning(f"{code} 超时（>{SUBPROCESS_TIMEOUT}秒），已跳过")
            timeout_count += 1
            continue

        try:
            result = queue.get(timeout=10)
        except Exception:
            logger.warning(f"{code} 子进程未返回结果，已跳过")
            continue
        if result["error"]:
            logger.error(f"{code} 拉取异常: {result['error']}")
        elif result["df"] is not None and not result["df"].empty:
            _save_daily(code, result["df"])
            success += 1

    logger.info(f"完成。成功: {success}, 跳过: {skip}, 超时: {timeout_count}, 总计: {total}")


# ============================================================
# 5. 数据加载
# ============================================================

def load_all_daily() -> pd.DataFrame:
    """加载全部已落地日线为一张大表"""
    files = sorted(cfg.DAILY_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("data/daily/ 下无数据，请先运行 fetch_all_daily()")

    dfs = []
    for f in tqdm(files, desc="加载本地数据"):
        code = f.stem
        df = pd.read_parquet(f)
        df["code"] = code
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    return full.sort_values(["date", "code"]).reset_index(drop=True)


def load_names() -> pd.DataFrame:
    """加载 A 股名称映射（仅 type=1 股票，排除指数）"""
    _bs_login()
    rs = bs.query_stock_basic()
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(rows, columns=["code", "name", "ipo_date", "out_date", "type", "status"])
    # 只保留 A 股（type=1 股票, status=1 上市中）
    df = df[(df["type"] == "1") & (df["status"] == "1")]
    df["code"] = df["code"].apply(_from_bs_code)
    return df[["code", "name"]]


def load_industries() -> pd.DataFrame:
    """加载行业分类
    Baostock 返回: [update_date, code, name, industry, source]
    """
    _bs_login()
    rs = bs.query_stock_industry()
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(rows)
    if df.empty or len(df.columns) < 4:
        return pd.DataFrame(columns=["code", "industry"])
    df = df.iloc[:, [1, 3]]
    df.columns = ["code", "industry"]
    df["code"] = df["code"].apply(_from_bs_code)
    df["industry"] = df["industry"].replace("", "未知")
    return df


def data_coverage_report() -> pd.DataFrame:
    files = sorted(cfg.DAILY_DIR.glob("*.parquet"))
    if not files:
        logger.warning("无本地数据")
        return pd.DataFrame()

    stats = []
    for f in files:
        df = pd.read_parquet(f, columns=["date"])
        stats.append({
            "code": f.stem,
            "start": df["date"].min(),
            "end":   df["date"].max(),
            "rows":  len(df),
        })

    report = pd.DataFrame(stats)
    logger.info(f"本地股票: {len(report)}, 行数: {report['rows'].sum():,}")
    return report


# ============================================================
# 6. 衍生计算
# ============================================================

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df.groupby("code")["close"].shift(1))
    return df


def add_ma(df: pd.DataFrame, windows=(5, 10, 20, 60)) -> pd.DataFrame:
    df = df.sort_values(["code", "date"])
    for w in windows:
        df[f"ma_{w}"] = df.groupby("code")["close"].transform(lambda x: x.rolling(w).mean())
    return df


from config import FINANCIAL_FIELDS

# ============================================================
# 7. 财务数据拉取
# ============================================================

# 查询函数映射
_FINANCE_QUERIES = {
    "profit":    bs.query_profit_data,
    "balance":   bs.query_balance_data,
    "cash_flow": bs.query_cash_flow_data,
    "growth":    bs.query_growth_data,
    "operation": bs.query_operation_data,
    "dupont":    bs.query_dupont_data,
}


def _query_finance_table(code: str, table: str, year: int, quarter: int) -> list:
    """查询单个财务表的一个季度（调用前需已登录）"""
    rs = _FINANCE_QUERIES[table](code=_to_bs_code(code), year=year, quarter=quarter)
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    return rows


def fetch_one_stock_financials(code: str, start_year: int = 2020) -> pd.DataFrame | None:
    """
    拉取单只股票全部财务数据

    返回 DataFrame，每行一个季度，列含六大类指标：
        code, pub_date, stat_date,
        roe_avg, np_margin, gp_margin, net_profit, eps_ttm, ...  (利润/盈利)
        current_ratio, quick_ratio, ...                           (偿债)
        ca_to_asset, cfo_to_np, ...                               (现金流)
        yoy_equity, yoy_ni, ...                                   (成长)
        nr_turn_ratio, inv_turn_ratio, ...                        (营运)
        dupont_roe, dupont_asset_turn, ...                        (杜邦)
    """
    from datetime import datetime
    current_year = datetime.today().year
    all_rows = []

    _bs_login()
    try:
        for year in range(start_year, current_year + 1):
            for quarter in range(1, 5):
                row_dict = {"code": code}

                for table_name, meta in FINANCIAL_FIELDS.items():
                    rows = _query_finance_table(code, table_name, year, quarter)
                    if rows:
                        r = rows[0]  # Baostock 每季度返回 1 行
                        row_dict["pub_date"] = r[1]
                        row_dict["stat_date"] = r[2]
                        # 数据从第 3 列开始
                        for i, col in enumerate(meta["columns"]):
                            row_dict[col] = _float_or_nan(r[3 + i])
                    else:
                        # 填入 NaN 占位
                        for col in meta["columns"]:
                            row_dict.setdefault(col, np.nan)

                # 只有至少有一个有效数据时才保留
                has_data = any(
                    not np.isnan(row_dict.get(c, np.nan))
                    for meta in FINANCIAL_FIELDS.values()
                    for c in meta["columns"]
                )
                if has_data:
                    all_rows.append(row_dict)

                time.sleep(cfg.REQUEST_DELAY * 0.5)
    finally:
        bs.logout()

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["stat_date"] = pd.to_datetime(df["stat_date"])
    df = df.drop_duplicates(subset="stat_date").sort_values("stat_date").reset_index(drop=True)
    return df


def _save_financials(code: str, df: pd.DataFrame):
    fpath = cfg.FINANCIAL_DIR / f"{code}.parquet"
    df.to_parquet(fpath, index=False)


def fetch_all_financials(
    codes: list | None = None,
    start_year: int = 2020,
    max_stocks: int = 0,
):
    """批量拉取财务数据（串行，避免 Baostock 会话冲突）"""
    from datetime import datetime as dt

    if codes is None:
        existing_codes = [f.stem for f in cfg.DAILY_DIR.glob("*.parquet")]
        codes = sorted(existing_codes)

    if max_stocks > 0:
        codes = codes[:max_stocks]

    total = len(codes)
    success = skip = fail = 0
    logger.info(f"拉取 {total} 只股票财务数据, {start_year} 至今")

    for code in tqdm(codes, desc="拉取财务数据"):
        # 检查是否已有最近季度的数据
        fpath = cfg.FINANCIAL_DIR / f"{code}.parquet"
        if fpath.exists():
            existing = pd.read_parquet(fpath)
            if len(existing) > 0:
                latest_stat = pd.to_datetime(existing["stat_date"].max())
                expected_latest = dt(dt.today().year, ((dt.today().month - 1) // 3) * 3 + 1, 1)
                if latest_stat >= pd.Timestamp(expected_latest - pd.DateOffset(months=6)):
                    skip += 1
                    continue

        try:
            df = fetch_one_stock_financials(code, start_year)
            if df is not None and not df.empty:
                _save_financials(code, df)
                success += 1
            else:
                skip += 1
        except Exception as e:
            logger.error(f"{code} 财务数据异常: {e}")
            fail += 1

    logger.info(f"财务数据完成。成功: {success}, 跳过: {skip}, 失败: {fail}, 总计: {total}")


def load_all_financials() -> pd.DataFrame:
    """加载全部已落地财务数据为一张大表"""
    files = sorted(cfg.FINANCIAL_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("data/financial/ 下无数据")

    dfs = []
    for f in tqdm(files, desc="加载财务数据"):
        df = pd.read_parquet(f)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    full["stat_date"] = pd.to_datetime(full["stat_date"])
    full["pub_date"] = pd.to_datetime(full["pub_date"])
    return full.sort_values(["code", "stat_date"]).reset_index(drop=True)


def add_atr(df: pd.DataFrame, window=14) -> pd.DataFrame:
    df = df.sort_values(["code", "date"])
    grp = df.groupby("code")
    prev_close = grp["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = grp["tr"].transform(lambda x: x.rolling(window).mean())
    return df
