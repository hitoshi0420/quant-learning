"""
风控模块 — 止损/波动率目标/行业上限/个股权重/换手率控制
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from dataclasses import dataclass
from collections import Counter
import numpy as np
import pandas as pd

from config import RISK_PARAMS


@dataclass
class RiskParams:
    max_single_weight: float = RISK_PARAMS["max_single_weight"]
    min_single_weight: float = RISK_PARAMS["min_single_weight"]
    max_industry_weight: float = RISK_PARAMS["max_industry_weight"]
    stop_loss: float = RISK_PARAMS["stop_loss"]
    portfolio_stop: float = RISK_PARAMS["portfolio_stop"]
    vol_target: float = RISK_PARAMS["vol_target"]
    min_cash_buffer: float = RISK_PARAMS["min_cash_buffer"]
    max_turnover: float = RISK_PARAMS["max_turnover"]


# ============================================================
# 1. 个股权重限制
# ============================================================

def cap_single_weights(portfolio: pd.DataFrame,
                       max_weight: float = 0.10,
                       min_weight: float = 0.01) -> pd.DataFrame:
    """对超过上限的个股降权，并重新分配"""
    result = portfolio.copy()
    excess = 0.0

    for i in result.index:
        if result.loc[i, "weight"] > max_weight:
            excess += result.loc[i, "weight"] - max_weight
            result.loc[i, "weight"] = max_weight

    if excess > 0:
        # 分配给未触上限的股票
        uncapped = result[result["weight"] < max_weight]
        if len(uncapped) > 0:
            per_stock = excess / len(uncapped)
            for i in uncapped.index:
                result.loc[i, "weight"] += per_stock

    # 剔除权重过小的
    result = result[result["weight"] >= min_weight]
    result["weight"] = result["weight"] / result["weight"].sum()

    return result


# ============================================================
# 2. 行业集中度控制
# ============================================================

def cap_industry_exposure(portfolio: pd.DataFrame,
                          industry_map: dict,
                          max_industry: float = 0.30) -> pd.DataFrame:
    """
    限制单一行业权重不超过 max_industry
    超限行业的最低得分/权重股被降权
    """
    result = portfolio.copy()
    result["industry"] = result["code"].map(industry_map).fillna("未知")

    # 计算行业权重
    ind_weights = result.groupby("industry")["weight"].sum()

    for ind, w in ind_weights.items():
        if w > max_industry and w > 0:
            scale = max_industry / w
            mask = result["industry"] == ind
            freed = result.loc[mask, "weight"].sum() * (1 - scale)
            result.loc[mask, "weight"] *= scale

            # 将释放的权重分配给其他行业
            other_mask = result["industry"] != ind
            if other_mask.any() and freed > 0:
                other_total = result.loc[other_mask, "weight"].sum()
                if other_total > 0:
                    result.loc[other_mask, "weight"] += freed * (
                        result.loc[other_mask, "weight"] / other_total
                    )

    result["weight"] = result["weight"] / result["weight"].sum()
    return result.drop(columns=["industry"])


# ============================================================
# 3. 止损检查
# ============================================================

def check_stop_loss(holdings: dict,
                    current_prices: dict,
                    stop_loss: float = -0.15) -> tuple[list[str], list[str]]:
    """
    检查每支持仓是否需要止损

    holdings: {code: {shares, cost_price, buy_date}}
    current_prices: {code: price}

    返回: (forced_sell_codes, warning_codes)
    """
    forced_sell = []
    warnings = []

    for code, h in holdings.items():
        # 兼容 avg_cost（paper_trading_engine）和 cost_price（旧格式）
        avg = h.get("avg_cost")
        cost = avg if avg is not None else h.get("cost_price", 0)
        if cost <= 0:
            continue
        current = current_prices.get(code, cost)
        pnl_pct = (current / cost - 1)

        if pnl_pct <= stop_loss:
            forced_sell.append(code)
        elif pnl_pct <= stop_loss * 0.7:  # 接近止损线
            warnings.append(code)

    return forced_sell, warnings


def check_portfolio_stop(portfolio_value: float,
                         peak_value: float,
                         portfolio_stop: float = -0.10) -> float:
    """
    组合回撤止损：返回建议仓位比例
    回撤超限 → 建议减至半仓
    """
    if peak_value <= 0:
        return 1.0

    drawdown = (portfolio_value / peak_value - 1)
    if drawdown <= portfolio_stop:
        return 0.50  # 减半仓
    elif drawdown <= portfolio_stop * 0.7:
        return 0.75  # 减至75%
    return 1.0


# ============================================================
# 4. 波动率目标
# ============================================================

def vol_target_scale(returns: pd.Series,
                     target_ann_vol: float = 0.15,
                     lookback: int = 60) -> float:
    """
    根据近期波动率调整仓位

    returns: 日收益序列
    返回: 仓位缩放系数 (≤1.0)
    """
    if len(returns) < lookback:
        return 1.0

    recent = returns.tail(lookback)
    realized_vol = recent.std() * np.sqrt(252)

    if np.isnan(realized_vol) or realized_vol <= 0:
        return 1.0

    scale = min(1.0, target_ann_vol / realized_vol)
    return scale


# ============================================================
# 5. 换手率控制
# ============================================================

def limit_turnover(new_weights: dict[str, float],
                   old_weights: dict[str, float],
                   max_turnover: float = 0.50) -> dict[str, float]:
    """
    限制换手率：新旧权重变化不超过 max_turnover

    换手率 = Σ|new_w - old_w| / 2
    """
    all_codes = set(new_weights) | set(old_weights)
    turnover = 0.0
    for code in all_codes:
        nw = new_weights.get(code, 0)
        ow = old_weights.get(code, 0)
        turnover += abs(nw - ow)
    turnover /= 2

    if turnover <= max_turnover:
        return new_weights

    # 温和过渡：70%旧权重 + 30%新权重
    scale = max_turnover / turnover if turnover > 0 else 1.0
    blended = {}
    for code in all_codes:
        nw = new_weights.get(code, 0)
        ow = old_weights.get(code, 0)
        blended[code] = ow + (nw - ow) * scale

    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended


# ============================================================
# 6. 一键风控流程
# ============================================================

def apply_risk_controls(portfolio: pd.DataFrame,
                        params: RiskParams = None,
                        industry_map: dict = None,
                        holdings: dict = None,
                        peak_value: float = None,
                        returns: pd.Series = None) -> pd.DataFrame:
    """
    对组合应用所有风控约束

    返回: 调整后的 portfolio DataFrame
    """
    if params is None:
        params = RiskParams()

    result = portfolio.copy()

    # 1. 个股权重上限
    result = cap_single_weights(result,
                                params.max_single_weight,
                                params.min_single_weight)

    # 2. 行业集中度
    if industry_map and len(industry_map) > 0:
        result = cap_industry_exposure(result, industry_map,
                                       params.max_industry_weight)

    # 3. 止损检查（仅报告，不修改权重）
    if holdings:
        if "close" in result.columns:
            price_dict = dict(zip(result["code"], result["close"]))
        else:
            price_dict = {}
        # 持仓代码不在新组合中 → 不设 price_dict，check_stop_loss 会用 cost 兜底
        # （设 0 会导致 pnl_pct = -1 错误触发止损）
        forced, warnings = check_stop_loss(holdings, price_dict, params.stop_loss)
        if forced:
            print(f"  [止损] 强制卖出: {forced}")
        if warnings:
            print(f"  [预警] 接近止损线: {warnings}")

    # 4. 波动率缩放（保留现金仓位，不再归一化回 1.0）
    if returns is not None:
        scale = vol_target_scale(returns, params.vol_target)
        if scale < 1.0:
            result["weight"] = result["weight"] * scale
            # 现金仓位 = 1 - sum(weights)，由调用方处理
            print(f"  [波动率] 仓位缩放: {scale:.0%}，现金占比: {(1 - result['weight'].sum()):.1%}")

    return result
