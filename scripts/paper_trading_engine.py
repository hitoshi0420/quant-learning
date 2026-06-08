"""
纸交易引擎 — 虚拟资金 + 真实股市规则 + v4.0 策略信号
纯虚拟，不涉及真实资金和真实交易
"""

import json
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Callable
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REC_CACHE_FILE = ROOT / "data" / "last_recommendation.json"

# 推荐进度跟踪（线程安全）
_rec_lock = threading.RLock()  # 可重入锁，避免死锁
_rec_status: dict = {
    "running": False,
    "step": "",
    "progress": 0,
    "message": "",
    "result": None,
    "error": None,
}


def _load_cached_recommendation():
    # 返回 Optional[dict]，兼容 Python 3.9
    """加载缓存的推荐结果，当日有效，或最近交易日数据未更新时沿用"""
    if not REC_CACHE_FILE.exists():
        return None
    try:
        with open(REC_CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cache_date = cached.get("target_date", "")
        today = datetime.now().strftime("%Y-%m-%d")
        # 缓存日期是今天 → 直接使用
        if cache_date == today:
            return cached
        # 缓存日期匹配数据中最新的交易日 → 数据未更新，沿用
        latest_date = _get_latest_data_date()
        if latest_date and cache_date == latest_date:
            return cached
    except Exception as e:
        print(f"[paper_trading] 加载缓存推荐失败: {e}")
    return None


def _get_latest_data_date():
    # 返回 Optional[str]，兼容 Python 3.9
    """获取本地数据中最新的日期"""
    daily_dir = ROOT / "data" / "daily"
    if not daily_dir.exists():
        return None
    try:
        import pandas as pd
        dates = set()
        import random
        files = list(daily_dir.glob("*.parquet"))
        samples = random.sample(files, min(20, len(files)))
        for fp in samples:
            df = pd.read_parquet(fp, columns=["date"])
            dates.add(str(df["date"].max())[:10])
        return max(dates) if dates else None
    except Exception as e:
        print(f"[paper_trading] 获取最新数据日期失败: {e}")
        return None


def _save_cached_recommendation(result: dict):
    """保存推荐结果到缓存文件"""
    if not result.get("success"):
        return
    import os
    REC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REC_CACHE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REC_CACHE_FILE)


def get_rec_status() -> dict:
    with _rec_lock:
        return dict(_rec_status)


def _reset_rec_status():
    with _rec_lock:
        _rec_status["running"] = True
        _rec_status["step"] = ""
        _rec_status["progress"] = 0
        _rec_status["message"] = ""
        _rec_status["result"] = None
        _rec_status["error"] = None


def _update_rec_status(step: str, progress: int, message: str):
    with _rec_lock:
        _rec_status["step"] = step
        _rec_status["progress"] = progress
        _rec_status["message"] = message

from config import COMMISSION_RATE, STAMP_DUTY, SLIPPAGE
MIN_SHARES = 100           # 最少100股（1手）
MIN_COMMISSION = 5.0       # 最低佣金5元


class PaperTradingEngine:
    """纸交易引擎 — 虚拟账户 + 真实规则"""

    def __init__(self, data_file: str = "paper_account.json"):
        self.data_file = ROOT / "data" / data_file
        self.account = self._load()

    # ============================================================
    # 账户管理
    # ============================================================

    def _default_account(self) -> dict:
        return {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "balance": 0.0,           # 可用现金
            "total_recharged": 0.0,   # 累计充值
            "positions": {},          # {code: {shares, avg_cost, buy_date, buy_price}}
            "transactions": [],       # [{date, action, code, shares, price, amount, commission, stamp, net}]
            "daily_snapshots": [],    # [{date, total_value, cash, market_value, n_positions}]
            "frozen_shares": {},      # {code: shares} T+1 冻结（今日买入不可卖）
        }

    def _load(self) -> dict:
        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[paper_trading] 加载账户文件失败 ({self.data_file}): {e}，重置为默认账户")
        return self._default_account()

    def save(self):
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.account, f, ensure_ascii=False, indent=2)

    def recharge(self, amount: float) -> dict:
        """充值虚拟币"""
        if amount <= 0:
            return {"success": False, "message": "充值金额必须大于0"}
        self.account["balance"] += amount
        self.account["total_recharged"] += amount
        self.save()
        return {
            "success": True,
            "message": f"成功充值 ¥{amount:,.2f}",
            "balance": self.account["balance"],
            "total_recharged": self.account["total_recharged"],
        }

    def get_account_summary(self, prices: dict = None, prev_prices: dict = None) -> dict:
        """账户摘要（支持计算当日盈亏）"""
        positions = self.account["positions"]
        market_value = 0.0
        today_pnl_total = 0.0
        prev_market_value = 0.0
        position_list = []

        for code, pos in positions.items():
            price = prices.get(code, pos["avg_cost"]) if prices else pos["avg_cost"]
            prev_price = prev_prices.get(code, price) if prev_prices else price
            mkt_val = pos["shares"] * price
            prev_mkt_val = pos["shares"] * prev_price
            pnl = mkt_val - pos["shares"] * pos["avg_cost"]
            pnl_pct = (price / pos["avg_cost"] - 1) if pos["avg_cost"] > 0 else 0
            today_pnl = mkt_val - prev_mkt_val
            pct_change = (price / prev_price - 1) * 100 if prev_price > 0 else 0
            market_value += mkt_val
            prev_market_value += prev_mkt_val
            today_pnl_total += today_pnl
            position_list.append({
                "code": code,
                "shares": pos["shares"],
                "avg_cost": round(pos["avg_cost"], 2),
                "current_price": round(price, 2),
                "prev_price": round(prev_price, 2),
                "pct_change": round(pct_change, 2),
                "market_value": round(mkt_val, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "today_pnl": round(today_pnl, 2),
                "buy_date": pos.get("buy_date", ""),
                "frozen": code in self.account.get("frozen_shares", {}),
            })

        total_value = self.account["balance"] + market_value
        total_pnl = total_value - self.account["total_recharged"]

        return {
            "balance": round(self.account["balance"], 2),
            "market_value": round(market_value, 2),
            "total_value": round(total_value, 2),
            "total_recharged": round(self.account["total_recharged"], 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / max(self.account["total_recharged"], 1) * 100, 2),
            "today_pnl": round(today_pnl_total, 2),
            "n_positions": len(positions),
            "positions": position_list,
        }

    # ============================================================
    # 交易
    # ============================================================

    def buy(self, code: str, shares: int, price: float, trade_date: str = None) -> dict:
        """
        买入股票
        - shares: 股数（会被调整为100的整数倍）
        - price: 当前市价
        - trade_date: 交易日期（用于T+1冻结）
        """
        if shares < MIN_SHARES:
            return {"success": False, "message": f"最少买入 {MIN_SHARES} 股"}
        if price <= 0:
            return {"success": False, "message": "价格无效"}

        # 调整为100的整数倍
        shares = (shares // MIN_SHARES) * MIN_SHARES
        if shares < MIN_SHARES:
            return {"success": False, "message": f"最少买入 {MIN_SHARES} 股"}

        actual_price = price * (1 + SLIPPAGE)
        cost = shares * actual_price
        commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
        total_cost = cost + commission

        if total_cost > self.account["balance"]:
            max_shares = int(self.account["balance"] / (actual_price * (1 + COMMISSION_RATE)) / MIN_SHARES) * MIN_SHARES
            if max_shares < MIN_SHARES:
                return {"success": False, "message": f"资金不足，需要 ¥{total_cost:,.2f}，可用 ¥{self.account['balance']:,.2f}"}
            shares = max_shares
            cost = shares * actual_price
            commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
            total_cost = cost + commission

        self.account["balance"] -= total_cost

        # 更新持仓（avg_cost 含佣金，确保 P&L 计算准确）
        per_share_cost = total_cost / shares
        if code in self.account["positions"]:
            pos = self.account["positions"][code]
            total_shares = pos["shares"] + shares
            pos["avg_cost"] = (pos["avg_cost"] * pos["shares"] + per_share_cost * shares) / total_shares
            pos["shares"] = total_shares
        else:
            self.account["positions"][code] = {
                "shares": shares,
                "avg_cost": float(per_share_cost),
                "buy_date": trade_date or datetime.now().strftime("%Y-%m-%d"),
                "buy_price": float(price),
            }

        # T+1 冻结
        if trade_date:
            if "frozen_shares" not in self.account:
                self.account["frozen_shares"] = {}
            self.account["frozen_shares"][code] = self.account["frozen_shares"].get(code, 0) + shares

        # 记录交易
        txn = {
            "date": trade_date or datetime.now().strftime("%Y-%m-%d"),
            "action": "买入",
            "code": code,
            "shares": shares,
            "price": round(float(price), 2),
            "actual_price": round(float(actual_price), 2),
            "amount": round(float(cost), 2),
            "commission": round(float(commission), 2),
            "stamp": 0,
            "net": round(float(-total_cost), 2),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        self.account["transactions"].append(txn)
        self.save()

        return {
            "success": True,
            "message": f"买入 {code} {shares}股 @ ¥{price:.2f}，花费 ¥{total_cost:,.2f}",
            "transaction": txn,
            "balance": round(self.account["balance"], 2),
        }

    def sell(self, code: str, shares: int, price: float, trade_date: str = None) -> dict:
        """
        卖出股票
        - 自动检查T+1冻结
        """
        if code not in self.account["positions"]:
            return {"success": False, "message": f"未持有 {code}"}

        pos = self.account["positions"][code]

        # T+1 检查
        frozen = self.account.get("frozen_shares", {}).get(code, 0)
        available = pos["shares"] - frozen
        if available < MIN_SHARES:
            return {"success": False, "message": f"{code} 可用 {available} 股（{frozen}股T+1冻结中），不足 {MIN_SHARES} 股"}

        if shares > available:
            shares = (available // MIN_SHARES) * MIN_SHARES
        else:
            shares = (shares // MIN_SHARES) * MIN_SHARES

        if shares < MIN_SHARES:
            return {"success": False, "message": f"最少卖出 {MIN_SHARES} 股"}

        actual_price = price * (1 - SLIPPAGE)
        proceeds = shares * actual_price
        commission = max(proceeds * COMMISSION_RATE, MIN_COMMISSION)
        stamp = proceeds * STAMP_DUTY
        net_proceeds = proceeds - commission - stamp

        self.account["balance"] += net_proceeds

        # 更新持仓
        pos["shares"] -= shares
        if pos["shares"] <= 0:
            del self.account["positions"][code]
            if code in self.account.get("frozen_shares", {}):
                del self.account["frozen_shares"][code]

        pnl = net_proceeds - shares * pos["avg_cost"]

        txn = {
            "date": trade_date or datetime.now().strftime("%Y-%m-%d"),
            "action": "卖出",
            "code": code,
            "shares": shares,
            "price": round(float(price), 2),
            "actual_price": round(float(actual_price), 2),
            "amount": round(float(proceeds), 2),
            "commission": round(float(commission), 2),
            "stamp": round(float(stamp), 2),
            "net": round(float(net_proceeds), 2),
            "pnl": round(float(pnl), 2),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        self.account["transactions"].append(txn)
        self.save()

        return {
            "success": True,
            "message": f"卖出 {code} {shares}股 @ ¥{price:.2f}，回收 ¥{net_proceeds:,.2f}",
            "transaction": txn,
            "pnl": round(float(pnl), 2),
            "balance": round(self.account["balance"], 2),
        }

    # ============================================================
    # 每日快照
    # ============================================================

    def take_snapshot(self, prices: dict, snap_date: str = None):
        """记录每日净值快照"""
        if snap_date is None:
            snap_date = datetime.now().strftime("%Y-%m-%d")

        # 检查今天是否已有快照
        for s in self.account["daily_snapshots"]:
            if s["date"] == snap_date:
                return s

        market_value = 0.0
        for code, pos in self.account["positions"].items():
            price = prices.get(code, pos["avg_cost"])
            market_value += pos["shares"] * price

        total = self.account["balance"] + market_value

        snap = {
            "date": snap_date,
            "total_value": round(total, 2),
            "cash": round(self.account["balance"], 2),
            "market_value": round(market_value, 2),
            "n_positions": len(self.account["positions"]),
        }
        self.account["daily_snapshots"].append(snap)

        # 只保留最近365天
        if len(self.account["daily_snapshots"]) > 365:
            self.account["daily_snapshots"] = self.account["daily_snapshots"][-365:]

        # 清除 T+1 冻结：只清除买入日期严格早于快照日期的冻结
        if "frozen_shares" in self.account:
            remaining = {}
            for code, shares in list(self.account["frozen_shares"].items()):
                pos = self.account["positions"].get(code)
                if pos and pos.get("buy_date", "") >= snap_date:
                    remaining[code] = shares
            self.account["frozen_shares"] = remaining

        self.save()
        return snap

    def get_daily_snapshots(self, days: int = 90) -> list:
        """获取最近N天的净值快照"""
        snaps = self.account.get("daily_snapshots", [])
        return snaps[-days:] if len(snaps) > days else snaps

    # ============================================================
    # 交易记录
    # ============================================================

    def get_transactions(self, limit: int = 50) -> list:
        """获取最近交易记录"""
        txns = self.account.get("transactions", [])
        return list(reversed(txns[-limit:]))

    # ============================================================
    # 策略推荐（调用 v4.0）
    # ============================================================

    def get_recommendations(self, top_n: int = 10, progress: Callable = None) -> dict:
        """调用v4.0多策略引擎获取推荐股票（含详细理由）"""
        import sys as _sys
        import traceback
        _scripts_dir = str(Path(__file__).resolve().parent)
        if _scripts_dir not in _sys.path:
            _sys.path.insert(0, _scripts_dir)
        try:
            from live_engine import LiveEngine
            engine = LiveEngine()

            def step(name, pct, msg):
                if progress:
                    progress(name, pct, msg)

            step("load", 5, "加载市场数据 + 财务数据...")
            engine.load_data(lookback_days=90)

            step("factors", 20, "计算因子截面（日频+财务）...")
            engine.compute_factors()

            step("ic", 40, "滚动IC分析 + 因子选择（三簇并行）...")
            engine.estimate_ic()

            step("score", 65, "三簇独立打分...")
            engine.score()

            step("build", 80, "多簇集成 + 组合优化 + 风控...")
            engine.build()

            if engine.portfolio is None or len(engine.portfolio) == 0:
                return {"success": False, "message": "无推荐结果"}

            step("reason", 90, "因子归因分析 + 生成推荐理由...")
            reasons = engine._analyze_pick_reasons()

            picks = []
            for _, row in engine.portfolio.head(top_n).iterrows():
                code = row["code"]
                reason = reasons.get(code, {})
                picks.append({
                    "code": code,
                    "name": engine.name_map.get(code, ""),
                    "industry": engine.ind_map.get(code, "未知"),
                    "close": round(float(row.get("close", 0)), 2),
                    "score": round(float(row.get("score", 0)), 3),
                    "weight": round(float(row.get("weight", 0)), 3),
                    "cluster": row.get("cluster", ""),
                    "reason": reason.get("summary", ""),
                    "style": reason.get("style", ""),
                    "strategy_desc": reason.get("strategy_desc", ""),
                    "top_factors": reason.get("top_factors", []),
                    "cluster_rank": reason.get("cluster_rank", ""),
                    "percentile": int(reason.get("percentile", 50)),
                })

            step("done", 100, f"生成 {len(picks)} 只推荐股票")

            return {
                "success": True,
                "target_date": engine.target_date.strftime("%Y-%m-%d"),
                "picks": picks,
            }
        except Exception as e:
            return {"success": False, "message": f"策略引擎错误: {e}\n{traceback.format_exc()[-300:]}"}

    def get_recommendations_async(self, top_n: int = 10):
        """异步获取推荐（后台线程），通过 get_rec_status() 跟踪进度"""
        global _rec_status
        with _rec_lock:
            if _rec_status["running"]:
                return {"success": False, "message": "推荐计算已在运行中"}
            _reset_rec_status()

        def run():
            try:
                result = self.get_recommendations(
                    top_n=top_n,
                    progress=lambda step, pct, msg: _update_rec_status(step, pct, msg),
                )
                with _rec_lock:
                    _rec_status["result"] = result
                    if result.get("success"):
                        _save_cached_recommendation(result)
            except Exception as e:
                with _rec_lock:
                    _rec_status["error"] = str(e)
            finally:
                with _rec_lock:
                    _rec_status["running"] = False

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return {"success": True, "message": "推荐计算已启动"}


# ============================================================
# 多用户引擎管理
# ============================================================

_engines: dict[int, PaperTradingEngine] = {}
_engines_lock = threading.Lock()


def get_engine(user_id: int = 0) -> PaperTradingEngine:
    """获取用户专属的纸交易引擎（user_id=0 为默认共享账户）"""
    with _engines_lock:
        if user_id not in _engines:
            data_file = "paper_account.json" if user_id == 0 else f"paper_account_{user_id}.json"
            _engines[user_id] = PaperTradingEngine(data_file=data_file)
        return _engines[user_id]
