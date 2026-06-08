"""
每日首次登录自动同步模块
- 检查是否今日已同步
- 增量补拉日线数据
- 更新股票名称和行业分类
- 支持后台线程非阻塞同步
"""

import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable

import pandas as pd
import baostock as bs
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_fetcher import _to_bs_code, _parse_row, COLUMN_MAP, _float_or_nan

DAILY_DIR = ROOT / "data" / "daily"
FINANCIAL_DIR = ROOT / "data" / "financial"
DATA_DIR = ROOT / "data"
STATE_FILE = DATA_DIR / ".sync_state.json"
SYNC_TIMES_FILE = DATA_DIR / ".stock_sync_times.json"
_sync_times_lock = threading.Lock()
_stock_sync_times: dict = {}

DAILY_FIELDS = (
    "date,open,high,low,close,preclose,volume,amount,"
    "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
)


def _safe_write_parquet(df: pd.DataFrame, fpath: Path):
    """原子写入 parquet：先写临时文件，再用 os.replace 原子替换
    os.replace 在所有平台（含 Windows）保证原子性，防止并发读取到半写入文件"""
    import os
    tmp = fpath.with_suffix(".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, fpath)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _safe_login():
    """登录 Baostock，检查返回值，失败时记录错误并重新抛出"""
    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"Baostock 登录失败: {lg.error_msg}")
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")
    return lg


# ============================================================
# 超时保护工具
# ============================================================

def _run_with_timeout(func: Callable, timeout: float, desc: str, *args, **kwargs):
    """在独立线程中运行函数，超时则记录警告并返回 None。
    注意：超时后原线程继续在后台运行（daemon），但主流程不等待。"""
    result = [None]
    exception = [None]

    def _target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.warning(f"{desc} 超时({timeout}s)，跳过此步骤继续后续同步")
        return None
    if exception[0]:
        raise exception[0]
    return result[0]


def _load_state() -> dict:
    """加载同步状态"""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[daily_sync] 读取同步状态失败: {e}")
    return {}


def _save_state(state: dict):
    """保存同步状态"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_sync_times() -> dict:
    """加载每只股票的最后同步时间"""
    if SYNC_TIMES_FILE.exists():
        try:
            return json.loads(SYNC_TIMES_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[daily_sync] 读取同步时间文件失败: {e}")
    return {}


def _save_sync_times(times: dict):
    """保存每只股票的最后同步时间"""
    SYNC_TIMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYNC_TIMES_FILE.write_text(json.dumps(times, ensure_ascii=False, indent=2), encoding="utf-8")


def _mark_stock_synced(code: str, data_type: str):
    """标记某只股票的某类数据已成功同步"""
    global _stock_sync_times
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _sync_times_lock:
        if code not in _stock_sync_times:
            _stock_sync_times[code] = {}
        _stock_sync_times[code][data_type] = now_str


def get_stock_sync_time(code: str) -> dict:
    """获取某只股票的最后同步时间（供仪表盘调用）"""
    global _stock_sync_times
    with _sync_times_lock:
        if not _stock_sync_times:
            _stock_sync_times.update(_load_sync_times())
        return dict(_stock_sync_times.get(code, {}))


def get_latest_trading_day() -> Optional[str]:
    """从 Baostock 获取最近一个交易日，返回 YYYY-MM-DD"""
    today = datetime.today()
    # 周末/节假日回退到最近周五
    wd = today.weekday()
    if wd >= 5:
        today = today - timedelta(days=wd - 4)
    lg = bs.login()
    if lg.error_code != "0":
        bs.logout()
        return today.strftime("%Y-%m-%d")

    try:
        rs = bs.query_trade_dates(
            start_date=(today - timedelta(days=14)).strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
        )
        dates = []
        while (rs.error_code == "0") and rs.next():
            row = rs.get_row_data()
            if row[0] == "1":
                dates.append(row[1])
        return max(dates) if dates else today.strftime("%Y-%m-%d")
    finally:
        bs.logout()


def scan_outdated_stocks() -> tuple[list[str], str]:
    """
    扫描本地数据，找出日线落后于最新交易日的股票
    返回: (需要更新的股票代码列表, 最新交易日)
    """
    latest_trading = get_latest_trading_day()
    if latest_trading is None:
        return [], ""

    target_dt = pd.to_datetime(latest_trading)
    files = sorted(DAILY_DIR.glob("*.parquet"))

    outdated = []
    for f in files:
        code = f.stem
        try:
            df = pd.read_parquet(f, columns=["date"])
            latest = str(df["date"].max())[:10]
            latest_dt = pd.to_datetime(latest)
            if latest_dt < target_dt:
                outdated.append(code)
        except Exception:
            outdated.append(code)

    logger.info(f"扫描结果: {len(files)} 只有本地数据, {len(outdated)} 只需更新")
    return outdated, latest_trading


def update_names():
    """更新股票名称映射表"""
    logger.info("更新股票名称映射...")
    _safe_login()
    try:
        rs = bs.query_stock_basic()
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())

        df = pd.DataFrame(rows, columns=["code", "name", "ipo_date", "out_date", "type", "status"])
        df = df[(df["type"] == "1") & (df["status"] == "1")]
        df["code_clean"] = df["code"].str.replace("sh.", "").str.replace("sz.", "")
        df[["code_clean", "name"]].to_csv(
            DATA_DIR / "stock_names.csv", index=False, encoding="utf-8-sig"
        )
        logger.info(f"股票名称表已更新: {len(df)} 只")
    finally:
        bs.logout()


def update_industries():
    """更新行业分类表"""
    logger.info("更新行业分类...")
    _safe_login()
    try:
        rs = bs.query_stock_industry()
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())

        df = pd.DataFrame(rows)
        df = df.iloc[:, [1, 3]]
        df.columns = ["code", "industry"]
        df["code"] = df["code"].str.replace("sh.", "").str.replace("sz.", "")
        df["industry"] = df["industry"].replace("", "未知")
        df.to_csv(DATA_DIR / "industry_map.csv", index=False, encoding="utf-8-sig")
        logger.info(f"行业分类已更新: {len(df)} 只, {df['industry'].nunique()} 个行业")
    finally:
        bs.logout()


DAILY_PER_STOCK_TIMEOUT = 25   # 单只日线超时(秒)
FIN_PER_STOCK_TIMEOUT = 60     # 单只财务超时(秒)
BATCH_ERROR_LIMIT = 50         # 连续失败上限，超过则终止当前批次
REQUEST_DELAY = 0.10           # 请求间隔(秒)，避免 Baostock 连接过载
RECONNECT_EVERY_N = 300        # 每 N 只股票重连一次 Baostock，防止 socket 耗尽
MAX_SOCKET_ERRORS = 10         # 连续 socket 错误上限，超限后重连


def sync_daily_incremental(codes: list[str], to_date: str) -> int:
    """
    增量补拉日线数据
    - 请求限速: 每只股票间隔 REQUEST_DELAY 秒，防止 Baostock 连接过载
    - Socket 恢复: 连续 socket 错误后自动重连 Baostock
    - 看门狗: 单只超时 > DAILY_PER_STOCK_TIMEOUT 秒自动跳过
    """
    if not codes:
        logger.info("所有股票日线已是最新，无需更新")
        return 0

    logger.info(f"开始增量更新 {len(codes)} 只股票 → {to_date}")

    with _sync_lock:
        _sync_status["daily_total"] = len(codes)
        _sync_status["daily_done"] = 0
        _sync_status["current_type"] = "daily"

    with _baostock_lock:
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"Baostock 登录失败: {lg.error_msg}")
            return 0

    updated = 0
    consecutive_errors = 0
    consecutive_socket_errors = 0
    try:
        for idx, code in enumerate(tqdm(codes, desc="增量更新日线")):
            with _sync_lock:
                _sync_status["current_stock"] = code
                _sync_status["current_stock_name"] = _get_stock_name(code)
                _sync_status["current_type"] = "daily"

            t_start = time.time()
            try:
                fpath = DAILY_DIR / f"{code}.parquet"
                old = pd.read_parquet(fpath)
                old_latest = str(old["date"].max())[:10]
                from_date = (pd.to_datetime(old_latest) + timedelta(days=1)).strftime("%Y-%m-%d")
                bs_code = _to_bs_code(code)

                # Baostock API 调用（加锁保护）
                with _baostock_lock:
                    if time.time() - t_start > DAILY_PER_STOCK_TIMEOUT:
                        logger.warning(f"{code} 等待 Baostock 锁超时，跳过")
                        with _sync_lock:
                            _sync_status["daily_done"] += 1
                        continue

                    rs = bs.query_history_k_data_plus(
                        bs_code, DAILY_FIELDS,
                        start_date=from_date, end_date=to_date,
                        frequency="d", adjustflag="2",
                    )
                    rows = []
                    while (rs.error_code == "0") and rs.next():
                        rows.append(rs.get_row_data())
                        if time.time() - t_start > DAILY_PER_STOCK_TIMEOUT:
                            break

                elapsed = time.time() - t_start
                if elapsed > DAILY_PER_STOCK_TIMEOUT:
                    with _sync_lock:
                        _sync_status["daily_done"] += 1
                    consecutive_errors += 1
                    continue

                if not rows:
                    with _sync_lock:
                        _sync_status["daily_done"] += 1
                    consecutive_errors = 0
                    continue

                new_df = pd.DataFrame([_parse_row(r) for r in rows])
                new_df["date"] = pd.to_datetime(new_df["date"])
                new_df = new_df[new_df["volume"] > 0]

                if new_df.empty:
                    with _sync_lock:
                        _sync_status["daily_done"] += 1
                    consecutive_errors = 0
                    continue

                combined = pd.concat([old, new_df], ignore_index=True)
                combined["date"] = pd.to_datetime(combined["date"])
                combined = combined.drop_duplicates(subset="date").sort_values("date")
                _safe_write_parquet(combined, fpath)
                updated += 1
                _mark_stock_synced(code, "daily")
                consecutive_errors = 0
                consecutive_socket_errors = 0

                with _sync_lock:
                    _sync_status["daily_done"] += 1

            except OSError as e:
                # Socket 错误（WinError 10038 等），尝试重连
                consecutive_errors += 1
                consecutive_socket_errors += 1
                with _sync_lock:
                    _sync_status["daily_done"] += 1
                if consecutive_socket_errors >= MAX_SOCKET_ERRORS:
                    logger.warning(f"连续 {consecutive_socket_errors} 次 socket 错误，重连 Baostock...")
                    with _baostock_lock:
                        try:
                            bs.logout()
                        except Exception:
                            pass
                        lg = bs.login()
                        if lg.error_code != "0":
                            logger.error("Baostock 重连失败，终止日线同步")
                            break
                    consecutive_socket_errors = 0
                    time.sleep(2)

            except Exception as e:
                with _sync_lock:
                    _sync_status["daily_done"] += 1
                consecutive_errors += 1
                logger.warning(f"{code} 日线更新失败: {e}")

            # 连续失败过多，终止
            if consecutive_errors >= BATCH_ERROR_LIMIT:
                logger.error(f"日线连续失败 {BATCH_ERROR_LIMIT} 次，终止日线同步")
                break

            # 定期重连，防止 socket 耗尽
            if (idx + 1) % RECONNECT_EVERY_N == 0 and idx > 0:
                with _baostock_lock:
                    try:
                        bs.logout()
                    except Exception:
                        pass
                    time.sleep(3)
                    lg = bs.login()
                    if lg.error_code != "0":
                        logger.error("Baostock 定期重连失败，终止日线同步")
                        break
                    consecutive_socket_errors = 0

            # 请求限速
            time.sleep(REQUEST_DELAY)

        logger.info(f"日线完成: {updated}/{len(codes)} 只")
    finally:
        with _baostock_lock:
            bs.logout()
        with _sync_lock:
            _sync_status["current_stock"] = ""
            _sync_status["current_stock_name"] = ""
            _sync_status["current_type"] = ""

    return updated


# ============================================================
# 财务数据增量同步
# ============================================================

from config import FINANCIAL_FIELDS

_FINANCE_QUERIES = {
    "profit":    bs.query_profit_data,
    "balance":   bs.query_balance_data,
    "cash_flow": bs.query_cash_flow_data,
    "growth":    bs.query_growth_data,
    "operation": bs.query_operation_data,
    "dupont":    bs.query_dupont_data,
}


def _get_expected_latest_quarter() -> tuple[int, int]:
    """返回 Baostock 预计已有数据的最新季度 (year, quarter)"""
    today = datetime.today()
    month = today.month
    # 财报通常在季度结束后1-2个月发布，当前季度可能还没数据
    if month <= 3:
        return today.year - 1, 4  # 上一年Q4
    elif month <= 6:
        return today.year, 1      # 今年Q1
    elif month <= 9:
        return today.year, 2      # 今年Q2
    else:
        return today.year, 3      # 今年Q3


def scan_outdated_financials() -> list[str]:
    """
    扫描财务数据，找出缺少最新季度的股票
    返回: 需要更新财务数据的股票代码列表
    """
    files = sorted(FINANCIAL_DIR.glob("*.parquet"))
    if not files:
        logger.info("无本地财务数据，跳过财务同步")
        return []

    exp_year, exp_quarter = _get_expected_latest_quarter()
    expected_latest = f"{exp_year}-{exp_quarter:02d}-01"

    outdated = []
    for f in files:
        code = f.stem
        try:
            df = pd.read_parquet(f)
            if "stat_date" not in df.columns or df.empty:
                outdated.append(code)
                continue
            latest = str(df["stat_date"].max())[:10]
            if latest < expected_latest:
                outdated.append(code)
        except Exception as e:
            print(f"[daily_sync] 财务扫描读取 {code} 失败: {e}")
            outdated.append(code)

    logger.info(f"财务扫描: {len(files)} 只本地, {len(outdated)} 只需更新 "
                f"(最新预期季度: {exp_year}Q{exp_quarter})")
    return outdated


def fetch_one_financial_incremental(code: str, existing_df: pd.DataFrame,
                                     bs_code: str) -> pd.DataFrame | None:
    """
    增量拉取单只股票缺失季度的财务数据
    已登录状态下调用
    """
    from_year = 2020
    if not existing_df.empty and "stat_date" in existing_df.columns:
        latest = pd.to_datetime(existing_df["stat_date"].max())
        from_year = latest.year

    current_year = datetime.today().year
    new_rows = []

    for year in range(from_year, current_year + 1):
        for quarter in range(1, 5):
            # 跳过已有数据的季度
            if not existing_df.empty and "stat_date" in existing_df.columns:
                q_date = f"{year}-{quarter*3:02d}-01"
                if q_date in existing_df["stat_date"].astype(str).values:
                    continue

            row_dict = {"code": code}
            has_data = False
            for table_name, meta in FINANCIAL_FIELDS.items():
                try:
                    rs = _FINANCE_QUERIES[table_name](code=bs_code, year=year, quarter=quarter)
                    rows = []
                    while (rs.error_code == "0") and rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        r = rows[0]
                        row_dict["pub_date"] = r[1]
                        row_dict["stat_date"] = r[2]
                        for i, col in enumerate(meta["columns"]):
                            row_dict[col] = _float_or_nan(r[3 + i])
                        has_data = True
                except Exception as e:
                    print(f"[daily_sync] 财务查询 {code} {table_name} 失败: {e}")

            if has_data:
                new_rows.append(row_dict)

    if not new_rows:
        return None

    result = pd.DataFrame(new_rows)
    if "pub_date" in result.columns:
        result["pub_date"] = pd.to_datetime(result["pub_date"])
    if "stat_date" in result.columns:
        result["stat_date"] = pd.to_datetime(result["stat_date"])
    return result


def sync_financials_incremental(codes: list[str]) -> int:
    """
    增量补拉财务数据
    - 看门狗: 单只超时 > FIN_PER_STOCK_TIMEOUT 秒跳过
    - Baostock 锁: 与日线线程互斥
    - 进度: 实时更新 current_stock / current_type
    """
    if not codes:
        logger.info("财务数据已全部就绪")
        return 0

    logger.info(f"开始增量更新财务数据: {len(codes)} 只")

    with _sync_lock:
        _sync_status["fin_total"] = len(codes)
        _sync_status["fin_done"] = 0
        _sync_status["current_type"] = "financial"

    with _baostock_lock:
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"Baostock 登录失败: {lg.error_msg}")
            return 0

    updated = 0
    consecutive_errors = 0
    try:
        for code in tqdm(codes, desc="增量更新财务"):
            with _sync_lock:
                _sync_status["current_stock"] = code
                _sync_status["current_type"] = "financial"

            t_start = time.time()
            try:
                fpath = FINANCIAL_DIR / f"{code}.parquet"
                existing = pd.read_parquet(fpath) if fpath.exists() else pd.DataFrame()
                bs_code = _to_bs_code(code)

                # Baostock API 调用（加锁保护）
                with _baostock_lock:
                    if time.time() - t_start > FIN_PER_STOCK_TIMEOUT:
                        logger.warning(f"{code} 等待 Baostock 锁超时，跳过")
                        with _sync_lock:
                            _sync_status["fin_done"] += 1
                        continue

                    new_data = fetch_one_financial_incremental(code, existing, bs_code)

                elapsed = time.time() - t_start
                if elapsed > FIN_PER_STOCK_TIMEOUT:
                    logger.warning(f"{code} 财务查询超时(>{FIN_PER_STOCK_TIMEOUT}s)，跳过")
                    with _sync_lock:
                        _sync_status["fin_done"] += 1
                    consecutive_errors += 1
                    continue

                if new_data is not None and not new_data.empty:
                    if not existing.empty:
                        combined = pd.concat([existing, new_data], ignore_index=True)
                        combined["stat_date"] = pd.to_datetime(combined["stat_date"])
                        combined = combined.drop_duplicates(subset="stat_date").sort_values("stat_date")
                    else:
                        combined = new_data
                    _safe_write_parquet(combined, fpath)
                    updated += 1
                    _mark_stock_synced(code, "financial")

                with _sync_lock:
                    _sync_status["fin_done"] += 1
                consecutive_errors = 0

                time.sleep(0.03)

            except Exception as e:
                with _sync_lock:
                    _sync_status["fin_done"] += 1
                consecutive_errors += 1
                logger.warning(f"{code} 财务更新失败: {e}")

            if consecutive_errors >= BATCH_ERROR_LIMIT:
                logger.error(f"财务连续失败 {BATCH_ERROR_LIMIT} 次，终止财务同步")
                break

        logger.info(f"财务完成: {updated}/{len(codes)} 只")
    finally:
        with _baostock_lock:
            bs.logout()
        with _sync_lock:
            _sync_status["current_stock"] = ""
            _sync_status["current_stock_name"] = ""
            _sync_status["current_type"] = ""

    return updated


_retry_timer: threading.Timer | None = None


def _schedule_retry(delay_minutes: int = 30):
    """数据未到位时，定时自动重试"""
    global _retry_timer

    if _retry_timer is not None:
        _retry_timer.cancel()

    logger.info(f"数据未到位，{delay_minutes} 分钟后自动重试...")

    def _do_retry():
        with _sync_lock:
            # 先快速判断是否还需要同步
            pass
        result = daily_check_and_sync()
        with _sync_lock:
            _sync_status["completed"] = True
            _sync_status["updated_count"] = result.get("updated_count", 0)
            _sync_status["fin_updated"] = result.get("fin_updated", 0)
            _sync_status["latest_trading"] = result.get("latest_trading", "")
            _sync_status["message"] = result.get("message", "")
            _sync_status["data_complete"] = result.get("data_complete", False)
        logger.info(f"自动重试完成: {result.get('message', '')}")

    _retry_timer = threading.Timer(delay_minutes * 60, _do_retry)
    _retry_timer.daemon = True
    _retry_timer.start()


def _verify_local_data_freshness(target_date: str, sample_size: int = 5) -> bool:
    """抽查本地文件是否包含目标日期的数据（防止 Baostock 延迟更新导致假同步）"""
    files = sorted(DAILY_DIR.glob("*.parquet"))
    if not files:
        return False

    import random
    samples = random.sample(files, min(sample_size, len(files)))
    for f in samples:
        try:
            df = pd.read_parquet(f, columns=["date"])
            if str(df["date"].max())[:10] < target_date:
                return False
        except Exception:
            return False
    return True


def needs_sync() -> bool:
    """判断是否需要同步"""
    state = _load_state()
    today_str = datetime.today().strftime("%Y-%m-%d")
    last_sync = state.get("last_sync_date", "")

    # 今天还没同步过
    if last_sync != today_str:
        return True

    # 检查是否有新的交易日（比如周一检查时周五的数据可能还没拉）
    latest_trading = get_latest_trading_day()
    last_trading = state.get("last_trading_day", "")
    if latest_trading and latest_trading != last_trading:
        return True

    # 上次同步时数据可能未到位（Baostock 延迟），抽查验证
    if not state.get("data_complete", True):
        return True
    if not _verify_local_data_freshness(latest_trading):
        return True

    return False


def daily_check_and_sync() -> dict:
    """
    每日首次登录检查并同步
    返回: {"synced": bool, "updated_count": int, "latest_trading": str, "message": str}
    """
    if not needs_sync():
        state = _load_state()
        return {
            "synced": False,
            "updated_count": 0,
            "latest_trading": state.get("last_trading_day", ""),
            "message": "今日已同步，数据已是最新",
        }

    logger.info("检测到需要同步，开始每日自动更新...")
    today_str = datetime.today().strftime("%Y-%m-%d")

    # 1. 更新名称和行业（带超时保护，跳过本地文件今天已存在的情况）
    _names_csv = DATA_DIR / "stock_names.csv"
    _ind_csv = DATA_DIR / "industry_map.csv"

    with _sync_lock:
        _sync_status["phase"] = "names"
    if _names_csv.exists():
        _names_mtime = datetime.fromtimestamp(_names_csv.stat().st_mtime)
        if _names_mtime.strftime("%Y-%m-%d") == today_str:
            logger.info("今日已更新股票名称，跳过")
        else:
            try:
                _run_with_timeout(update_names, 60, "股票名称更新")
            except Exception as e:
                logger.warning(f"名称更新失败: {e}")
    else:
        try:
            _run_with_timeout(update_names, 60, "股票名称更新")
        except Exception as e:
            logger.warning(f"名称更新失败: {e}")

    with _sync_lock:
        _sync_status["phase"] = "industries"
    if _ind_csv.exists():
        _ind_mtime = datetime.fromtimestamp(_ind_csv.stat().st_mtime)
        if _ind_mtime.strftime("%Y-%m-%d") == today_str:
            logger.info("今日已更新行业分类，跳过")
        else:
            try:
                _run_with_timeout(update_industries, 60, "行业分类更新")
            except Exception as e:
                logger.warning(f"行业更新失败: {e}")
    else:
        try:
            _run_with_timeout(update_industries, 60, "行业分类更新")
        except Exception as e:
            logger.warning(f"行业更新失败: {e}")

    # 2. 扫描待更新列表（带超时保护，失败用今日日期作为回退）
    with _sync_lock:
        _sync_status["phase"] = "scan"
    try:
        result = _run_with_timeout(scan_outdated_stocks, 90, "扫描过期股票")
        if result is None:
            outdated_codes, latest_trading = [], today_str
            logger.warning("扫描过期股票超时，使用今日日期作为回退")
        else:
            outdated_codes, latest_trading = result
    except Exception as e:
        logger.warning(f"扫描过期股票失败: {e}")
        outdated_codes, latest_trading = [], today_str

    fin_outdated = scan_outdated_financials()

    # 3. 日线和财务串行拉取（共用 Baostock 会话，避免互相登出）
    with _sync_lock:
        _sync_status["phase"] = "daily_financial"

    # 先日线
    daily_updated = sync_daily_incremental(outdated_codes, latest_trading)
    # 再财务
    fin_updated = sync_financials_incremental(fin_outdated)

    with _sync_lock:
        _sync_status["phase"] = "done"

    # 4. 验证数据是否真正到位（Baostock 可能延迟更新）
    data_complete = _verify_local_data_freshness(latest_trading)
    if not data_complete:
        logger.warning(f"Baostock 尚未更新 {latest_trading} 日线数据，稍后重试时将自动检测")

    # 5. 记录同步状态
    state = {
        "last_sync_date": today_str,
        "last_trading_day": latest_trading,
        "data_complete": data_complete,
        "daily_file_count": len(list(DAILY_DIR.glob("*.parquet"))),
        "financial_file_count": len(list(FINANCIAL_DIR.glob("*.parquet"))),
    }
    _save_state(state)
    _save_sync_times(_stock_sync_times)

    if not data_complete:
        message = f"同步完成: 日线 {daily_updated} 只 + 财务 {fin_updated} 只, Baostock 尚未更新 {latest_trading} 数据(30分钟后自动重试)"
        _schedule_retry()
    else:
        message = f"同步完成: 日线 {daily_updated} 只 + 财务 {fin_updated} 只, 最新交易日 {latest_trading}"
    logger.info(message)

    return {
        "synced": True,
        "updated_count": daily_updated,
        "fin_updated": fin_updated,
        "latest_trading": latest_trading,
        "data_complete": data_complete,
        "message": message,
    }


# ============================================================
# 后台线程同步（非阻塞，给 dashboard 用）
# ============================================================

_sync_lock = threading.Lock()
_baostock_lock = threading.Lock()  # 保护 Baostock 调用，防止并发冲突
_sync_status: dict = {
    "running": False,
    "completed": False,
    "phase": "idle",
    "updated_count": 0,
    "fin_updated": 0,
    "daily_total": 0,
    "daily_done": 0,
    "fin_total": 0,
    "fin_done": 0,
    "current_stock": "",       # 当前正在拉取的股票代码
    "current_stock_name": "",  # 当前股票名称
    "current_type": "",        # 当前拉取类型: daily / financial
    "latest_trading": "",
    "data_complete": True,     # 最新交易日数据是否已到位
    "message": "等待检查...",
    "start_time": None,
}


def _load_stock_names() -> dict:
    """从本地 CSV 加载代码→名称映射"""
    try:
        import pandas as pd
        df = pd.read_csv(DATA_DIR / "stock_names.csv", dtype=str)
        return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    except Exception:
        return {}

_stock_names_cache: dict | None = None

def _get_stock_name(code: str) -> str:
    global _stock_names_cache
    if _stock_names_cache is None:
        _stock_names_cache = _load_stock_names()
    return _stock_names_cache.get(code, "")


def get_sync_status() -> dict:
    """获取当前同步状态（线程安全）"""
    with _sync_lock:
        return dict(_sync_status)


def _run_sync_in_background():
    """后台线程执行同步"""
    global _sync_status
    with _sync_lock:
        _sync_status["running"] = True
        _sync_status["phase"] = "names"
        _sync_status["start_time"] = datetime.now().strftime("%H:%M:%S")
        _sync_status["message"] = "正在同步股票数据..."

    try:
        result = daily_check_and_sync()
        with _sync_lock:
            _sync_status["running"] = False
            _sync_status["completed"] = True
            _sync_status["updated_count"] = result["updated_count"]
            _sync_status["fin_updated"] = result.get("fin_updated", 0)
            _sync_status["latest_trading"] = result["latest_trading"]
            _sync_status["data_complete"] = result.get("data_complete", True)
            _sync_status["message"] = result["message"]
    except Exception as e:
        with _sync_lock:
            _sync_status["running"] = False
            _sync_status["completed"] = True
            _sync_status["message"] = f"同步异常: {e}"


def start_background_sync() -> dict:
    """
    如果今日需要同步，启动后台线程执行
    返回当前状态快照
    """
    with _sync_lock:
        if _sync_status["running"]:
            return dict(_sync_status)

    # 先快速检查是否需要同步（不加锁）
    if needs_sync():
        state = _load_state()
        latest = state.get("last_trading_day", "") or get_latest_trading_day() or ""
        t = threading.Thread(target=_run_sync_in_background, daemon=True)
        t.start()
        return {"running": True, "latest_trading": latest, "message": "检测到需要更新，后台同步已启动..."}
    else:
        state = _load_state()
        with _sync_lock:
            _sync_status["completed"] = True
            _sync_status["message"] = "今日已同步，数据已是最新"
            _sync_status["latest_trading"] = state.get("last_trading_day", "")
        return dict(_sync_status)


def trigger_manual_sync() -> dict:
    """手动触发同步（忽略 needs_sync 检查）"""
    with _sync_lock:
        if _sync_status["running"]:
            return {"ok": False, "message": "同步已在运行中"}

    t = threading.Thread(target=_run_sync_in_background, daemon=True)
    t.start()
    return {"ok": True, "message": "手动同步已启动"}


def get_data_summary() -> dict:
    """获取数据概览：日线时间范围 + 财务时间范围"""
    daily_files = sorted(DAILY_DIR.glob("*.parquet"))
    fin_files = sorted(FINANCIAL_DIR.glob("*.parquet"))

    d_earliest, d_latest = "", ""
    f_earliest, f_latest = "", ""

    if daily_files:
        try:
            import random
            samples = random.sample(daily_files, min(20, len(daily_files)))
            dates = []
            for fp in samples:
                df = pd.read_parquet(fp, columns=["date"])
                dates.append(str(df["date"].min())[:10])
                dates.append(str(df["date"].max())[:10])
            d_earliest = min(dates)
            d_latest = max(dates)
        except Exception:
            pass

    if fin_files:
        try:
            import random
            samples = random.sample(fin_files, min(20, len(fin_files)))
            dates = []
            for fp in samples:
                df = pd.read_parquet(fp, columns=["stat_date"])
                dates.append(str(df["stat_date"].min())[:10])
                dates.append(str(df["stat_date"].max())[:10])
            f_earliest = min(dates)
            f_latest = max(dates)
        except Exception:
            pass

    return {
        "daily_count": len(daily_files),
        "daily_earliest": d_earliest,
        "daily_latest": d_latest,
        "financial_count": len(fin_files),
        "financial_earliest": f_earliest,
        "financial_latest": f_latest,
    }


def _run_historical_sync(start_date: str, end_date: str):
    """后台拉取指定时间段的日线数据"""
    global _sync_status
    with _sync_lock:
        _sync_status["running"] = True
        _sync_status["phase"] = "daily"
        _sync_status["current_type"] = "daily"
        _sync_status["start_time"] = datetime.now().strftime("%H:%M:%S")
        _sync_status["message"] = f"正在拉取历史数据 {start_date} ~ {end_date}..."

    try:
        lg = bs.login()
        if lg.error_code != "0":
            with _sync_lock:
                _sync_status["running"] = False
                _sync_status["message"] = "Baostock 登录失败"
            return

        files = sorted(DAILY_DIR.glob("*.parquet"))
        codes = [f.stem for f in files]

        with _sync_lock:
            _sync_status["daily_total"] = len(codes)
            _sync_status["daily_done"] = 0

        updated = 0
        for i, code in enumerate(codes):
            try:
                with _sync_lock:
                    _sync_status["current_stock"] = code
                    _sync_status["current_stock_name"] = _get_stock_name(code)
                    _sync_status["daily_done"] = i + 1

                bs_code = _to_bs_code(code)
                with _baostock_lock:
                    rs = bs.query_history_k_data_plus(
                        bs_code, DAILY_FIELDS,
                        start_date=start_date, end_date=end_date,
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
                _safe_write_parquet(combined, fpath)
                updated += 1
                _mark_stock_synced(code, "daily")

                time.sleep(0.05)
            except Exception:
                pass

        with _sync_lock:
            _sync_status["running"] = False
            _sync_status["completed"] = True
            _sync_status["phase"] = "done"
            _sync_status["current_stock"] = ""
            _sync_status["current_stock_name"] = ""
            _sync_status["message"] = f"历史拉取完成: {updated}/{len(codes)} 只, {start_date}~{end_date}"

        logger.info(f"历史拉取完成: {updated} 只")
    finally:
        try:
            bs.logout()
        except Exception:
            pass
