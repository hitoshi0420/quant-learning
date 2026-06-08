"""
组合构建器 — 等权/ICIR加权/MaxSharpe/风险平价/最小方差
"""

import numpy as np
import pandas as pd
from enum import Enum
from scipy.optimize import minimize


class PortfolioMethod(Enum):
    EQUAL = "equal"
    ICIR_WEIGHTED = "ic_ir"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"


def _shrinkage_covariance(returns: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    """Ledoit-Wolf 简化版收缩估计（constant correlation 目标）"""
    sample_cov = np.cov(returns, rowvar=False)
    n = sample_cov.shape[0]
    diag_vals = np.diag(sample_cov)
    stds = np.sqrt(np.maximum(diag_vals, 1e-12))

    # 计算平均相关系数作为收缩目标
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = sample_cov / np.outer(stds, stds)
    np.fill_diagonal(corr, np.nan)
    avg_corr = np.nanmean(corr)
    if np.isnan(avg_corr) or avg_corr <= -1.0:
        avg_corr = 0.0
    avg_corr = max(avg_corr, 0.0)  # 约束为非负

    # Constant correlation target: 对角线=方差, 非对角线=avg_corr * std_i * std_j
    target = avg_corr * np.outer(stds, stds)
    np.fill_diagonal(target, diag_vals)

    # 自适应收缩强度（n 大时收缩更多）
    adaptive_shrinkage = max(shrinkage, n / (n + returns.shape[0]))
    return (1 - adaptive_shrinkage) * sample_cov + adaptive_shrinkage * target


def equal_weight(scores: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """等权分配"""
    selected = scores.head(top_n).copy()
    selected["weight"] = 1.0 / len(selected)
    return selected


def icir_weighted(scores: pd.DataFrame, factor_weights: dict = None, top_n: int = 30) -> pd.DataFrame:
    """按得分的平方根分配权重（IC_IR加权等风险贡献）"""
    selected = scores.head(top_n).copy()

    if "score" not in selected.columns:
        selected["weight"] = 1.0 / len(selected)
        return selected

    # 得分可能有负数，平移使最小值=0.1
    min_score = selected["score"].min()
    shifted = selected["score"] - min_score + 0.1

    # 平方根权重 → 更均衡
    raw_weights = np.sqrt(shifted)
    selected["weight"] = raw_weights / raw_weights.sum()
    return selected


def max_sharpe_weights(expected_returns: np.ndarray,
                       cov_matrix: np.ndarray,
                       max_w: float = 0.10,
                       min_w: float = 0.01) -> np.ndarray:
    """Max Sharpe 组合优化"""
    n = len(expected_returns)

    def neg_sharpe(w):
        pr = np.dot(w, expected_returns)
        pv = np.sqrt(max(np.dot(w, np.dot(cov_matrix, w)), 1e-10))
        return -pr / pv

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-8})

    if result.success:
        w = np.maximum(result.x, 0)
        s = w.sum()
        return w / s if s > 0 else x0
    return x0


def risk_parity_weights(cov_matrix: np.ndarray,
                        max_w: float = 0.10,
                        min_w: float = 0.01) -> np.ndarray:
    """风险平价：使每只股票对组合风险的边际贡献相等"""
    n = cov_matrix.shape[0]

    def risk_concentration(w):
        pv = np.sqrt(max(np.dot(w, np.dot(cov_matrix, w)), 1e-10))
        mrc = np.dot(cov_matrix, w) / pv   # 边际风险贡献
        rc = w * mrc                        # 绝对风险贡献
        target_rc = pv / n                  # 目标：每只股票贡献相同风险
        return np.sum((rc - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(risk_concentration, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-10})

    if result.success:
        w = np.maximum(result.x, 0)
        s = w.sum()
        return w / s if s > 0 else x0
    return x0


def min_variance_weights(cov_matrix: np.ndarray,
                         max_w: float = 0.10,
                         min_w: float = 0.01) -> np.ndarray:
    """最小方差组合"""
    n = cov_matrix.shape[0]

    def portfolio_variance(w):
        return np.dot(w, np.dot(cov_matrix, w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(portfolio_variance, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-8})

    if result.success:
        w = np.maximum(result.x, 0)
        s = w.sum()
        return w / s if s > 0 else x0
    return x0


def build_portfolio(scores: pd.DataFrame,
                    method: PortfolioMethod = PortfolioMethod.ICIR_WEIGHTED,
                    top_n: int = 30,
                    returns_df: pd.DataFrame = None,
                    factor_weights: dict = None,
                    constraints: dict = None) -> pd.DataFrame:
    """
    统一组合构建接口

    Args:
        scores: 已打分的截面数据 (含 'code', 'close', 'score')
        method: 组合构建方法
        top_n: 入选股票数
        returns_df: 历史收益数据 (wide format: date x code)，用于需要协方差的方法
        factor_weights: IC_IR权重字典
        constraints: {max_weight, min_weight, ...}

    Returns:
        DataFrame: [code, weight, ...] 按权重降序排列
    """
    if constraints is None:
        constraints = {"max_weight": 0.10, "min_weight": 0.01}

    selected_codes = scores.head(top_n)["code"].tolist()
    n = len(selected_codes)

    if method == PortfolioMethod.EQUAL:
        return equal_weight(scores, top_n)

    elif method == PortfolioMethod.ICIR_WEIGHTED:
        return icir_weighted(scores, factor_weights, top_n)

    elif method in (PortfolioMethod.MAX_SHARPE, PortfolioMethod.RISK_PARITY,
                    PortfolioMethod.MIN_VARIANCE):
        # 需要收益率协方差
        if returns_df is None or len(returns_df) < 12:
            # 降级为等权
            print(f"  [警告] {method.value} 需要收益率数据，降级为等权")
            return equal_weight(scores, top_n)

        # 取共同列，不足 top_n 时向下扩展
        common = [c for c in selected_codes if c in returns_df.columns]
        if len(common) < min(top_n, 5):
            return equal_weight(scores, top_n)
        while len(common) < top_n:
            remaining = scores[~scores["code"].isin(common)]
            extras = [c for c in remaining["code"].tolist() if c in returns_df.columns]
            if not extras:
                break
            common.append(extras[0])

        ret_values = returns_df[common].values
        cov = _shrinkage_covariance(ret_values)

        if method == PortfolioMethod.MAX_SHARPE:
            mu = ret_values.mean(axis=0)
            weights = max_sharpe_weights(mu, cov,
                                         constraints.get("max_weight", 0.10),
                                         constraints.get("min_weight", 0.01))
        elif method == PortfolioMethod.RISK_PARITY:
            weights = risk_parity_weights(cov,
                                          constraints.get("max_weight", 0.10),
                                          constraints.get("min_weight", 0.01))
        elif method == PortfolioMethod.MIN_VARIANCE:
            weights = min_variance_weights(cov,
                                           constraints.get("max_weight", 0.10),
                                           constraints.get("min_weight", 0.01))

        selected = scores[scores["code"].isin(common)].head(n).copy()
        weight_map = dict(zip(common, weights))
        selected["weight"] = selected["code"].map(weight_map).fillna(0)
        selected["weight"] = selected["weight"] / selected["weight"].sum()
        return selected.sort_values("weight", ascending=False)

    else:
        return equal_weight(scores, top_n)
