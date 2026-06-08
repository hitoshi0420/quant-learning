"""
滚动IC估计器 — 动态IC_IR + 因子去相关 + 显著性筛选
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def compute_ic_analysis(factor_table: pd.DataFrame,
                        factor_cols: list[str],
                        lookback_months: int = 36,
                        decay_halflife: int = 12) -> pd.DataFrame:
    """
    对每个因子计算滚动IC分析（指数衰减加权）

    返回 DataFrame: [factor, ic_mean, ic_std, ic_ir, t_stat, n_months]
    """
    all_dates = sorted(factor_table["date"].unique())
    results = []

    for col in tqdm(factor_cols, desc="IC分析"):
        ic_list = []
        ic_dates = []

        for date, grp in factor_table.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) < 30:
                continue
            try:
                ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
                if not np.isnan(ic):
                    ic_list.append(ic)
                    ic_dates.append(date)
            except Exception as e:
                print(f"[ic_estimator] spearmanr failed for {col} on {date}: {e}")

        if len(ic_list) < 6:
            continue

        ic_arr = np.array(ic_list)
        ic_dates_arr = np.array(ic_dates)

        # 如果指定了 lookback，只取最近的
        if lookback_months > 0 and len(ic_arr) > lookback_months:
            ic_arr = ic_arr[-lookback_months:]
            ic_dates_arr = ic_dates_arr[-lookback_months:]

        # 指数衰减加权
        if decay_halflife > 0 and len(ic_arr) > 1:
            ages = len(ic_arr) - 1 - np.arange(len(ic_arr))  # 0=最新
            lambda_decay = np.log(2) / decay_halflife
            weights = np.exp(-lambda_decay * ages)
            weights = weights / weights.sum()

            ic_mean = np.average(ic_arr, weights=weights)
            ic_std = np.sqrt(np.average((ic_arr - ic_mean) ** 2, weights=weights))
        else:
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr, ddof=1)

        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0

        results.append({
            "factor": col,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "t_stat": t_stat,
            "n_months": len(ic_arr),
        })

    return pd.DataFrame(results).sort_values("ic_ir", key=abs, ascending=False)


def compute_factor_correlation(factor_table: pd.DataFrame,
                               factor_cols: list[str]) -> pd.DataFrame:
    """计算因子间截面Spearman相关矩阵（pooled across all dates）"""
    all_corrs = []
    for _, grp in factor_table.groupby("date"):
        valid = grp[factor_cols].dropna()
        if len(valid) < 30:
            continue
        corr = valid.corr(method="spearman")
        all_corrs.append(corr)

    if not all_corrs:
        return pd.DataFrame(index=factor_cols, columns=factor_cols, data=0.0)

    # 取平均相关
    avg_corr = sum(all_corrs) / len(all_corrs)
    return avg_corr


def decorrelate_factors(ic_report: pd.DataFrame,
                        corr_matrix: pd.DataFrame,
                        threshold: float = 0.7) -> tuple[list[str], dict]:
    """
    去相关处理：
    1. 对 |corr| > threshold 的因子对，保留 IC_IR 绝对值更高者
    2. 对保留的因子按相关簇大小调整权重

    返回: (selected_factors, weight_multipliers)
    """
    factor_cols = [c for c in ic_report["factor"] if c in corr_matrix.columns]
    if len(factor_cols) <= 1:
        return factor_cols, {f: 1.0 for f in factor_cols}

    # 构建 IC_IR 绝对值查找表
    ic_map = {}
    for _, row in ic_report.iterrows():
        ic_map[row["factor"]] = abs(row["ic_ir"])

    # 贪心去相关
    removed = set()
    sorted_factors = sorted(factor_cols, key=lambda f: ic_map.get(f, 0), reverse=True)

    for i, f1 in enumerate(sorted_factors):
        if f1 in removed:
            continue
        for f2 in sorted_factors[i + 1:]:
            if f2 in removed:
                continue
            if f1 in corr_matrix.index and f2 in corr_matrix.columns:
                corr_val = abs(corr_matrix.loc[f1, f2])
                if corr_val > threshold:
                    removed.add(f2)

    selected = [f for f in sorted_factors if f not in removed]
    n_removed = len(removed)
    if n_removed > 0:
        print(f"  去相关: 移除 {n_removed} 个高相关因子 (相关>{threshold})")
        for f in sorted(removed):
            print(f"    - {f.replace('factor_', '')}")

    # 每个保留因子按相关簇大小调整权重
    weight_mult = {}
    for f in selected:
        if f in corr_matrix.index:
            n_correlated = sum(
                1 for other in corr_matrix.columns
                if other != f and abs(corr_matrix.loc[f, other]) > threshold
            )
            weight_mult[f] = 1.0 / (1 + n_correlated * 0.5)  # 温和惩罚
        else:
            weight_mult[f] = 1.0

    return selected, weight_mult


def get_factor_weights(ic_report: pd.DataFrame,
                       selected_factors: list[str],
                       weight_multipliers: dict,
                       max_group_weight: dict = None,
                       max_single_weight: float = 0.20) -> dict[str, float]:
    """
    计算最终因子权重，考虑：
    1. IC_IR 绝对值作为基础权重
    2. 去相关惩罚
    3. 分组权重上限
    4. 单因子权重上限

    返回: {factor_name: weight}
    """
    from factor_library import FACTOR_GROUPS, get_factor

    weights = {}
    total_ic = 0

    for f in selected_factors:
        row = ic_report[ic_report["factor"] == f]
        if len(row) == 0:
            continue
        w = abs(row["ic_ir"].values[0])
        w *= weight_multipliers.get(f, 1.0)
        weights[f] = w
        total_ic += w

    if total_ic == 0:
        return weights

    # 归一化
    for f in weights:
        weights[f] /= total_ic

    # 应用组权重上限
    if max_group_weight:
        group_weights = {g: 0.0 for g in FACTOR_GROUPS}
        for f, w in weights.items():
            lookup = f.replace("_raw", "") if f.endswith("_raw") else f
            fdef = get_factor(lookup)
            if fdef:
                group = fdef.group
                group_weights[group] = group_weights.get(group, 0) + w

        # 缩放超限组
        for g, gw in group_weights.items():
            limit = max_group_weight.get(g, 0.35)
            if gw > limit and gw > 0:
                scale = limit / gw
                for f in weights:
                    lookup = f.replace("_raw", "") if f.endswith("_raw") else f
                    fdef = get_factor(lookup)
                    if fdef and fdef.group == g:
                        weights[f] *= scale

        # 重新归一化
        total = sum(weights.values())
        if total > 0:
            for f in weights:
                weights[f] /= total

    # 应用单因子权重上限（迭代直到所有因子都不超限）
    for _ in range(5):  # 最多5轮，防止极端情况
        overflow = False
        for f in list(weights.keys()):
            if weights[f] > max_single_weight:
                overflow = True
                excess = weights[f] - max_single_weight
                weights[f] = max_single_weight
                others = [of for of in weights if of != f]
                if others:
                    extra = excess / len(others)
                    for of in others:
                        weights[of] += extra
        if not overflow:
            break

    return weights


def run_ic_pipeline(factor_table: pd.DataFrame,
                    min_ic_ir: float = None,
                    correlation_threshold: float = 0.7,
                    max_group_weight: dict = None) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """
    一键运行IC分析流水线：
    1. IC分析
    2. 因子相关矩阵
    3. 去相关
    4. 计算最终权重

    返回: (ic_report, selected_factors, final_weights)
    """
    from config import IC_LOOKBACK_MONTHS, IC_DECAY_HALFLIFE, MIN_IC_IR_THRESHOLD
    if min_ic_ir is None:
        min_ic_ir = MIN_IC_IR_THRESHOLD

    factor_cols = [c for c in factor_table.columns if c.startswith("factor_")
                   and c != "factor_size"]

    # Step 1: IC分析
    print("IC分析...")
    ic_report = compute_ic_analysis(factor_table, factor_cols,
                                     lookback_months=IC_LOOKBACK_MONTHS,
                                     decay_halflife=IC_DECAY_HALFLIFE)

    # 筛选显著因子
    significant = ic_report[abs(ic_report["ic_ir"]) >= min_ic_ir]
    if len(significant) < 5:
        significant = ic_report.head(8)
    sig_factors = significant["factor"].tolist()

    print(f"  显著因子 ({len(sig_factors)} 个, |IC_IR|≥{min_ic_ir}):")
    for _, row in significant.iterrows():
        name = row["factor"].replace("factor_", "")
        print(f"    {name:25s} IC_IR={row['ic_ir']:+.3f}  T={row['t_stat']:+.2f}")

    # Step 2: 相关矩阵
    print("因子相关性分析...")
    corr_matrix = compute_factor_correlation(factor_table, sig_factors)

    # Step 3: 去相关
    selected, weight_mult = decorrelate_factors(significant, corr_matrix, correlation_threshold)

    # Step 4: 最终权重
    final_weights = get_factor_weights(significant, selected, weight_mult,
                                       max_group_weight=max_group_weight)

    print(f"\n  最终权重 ({len(selected)} 个因子):")
    from factor_library import get_factor as _get_factor
    for f in sorted(selected, key=lambda x: final_weights.get(x, 0), reverse=True):
        w = final_weights.get(f, 0)
        # 去除 _raw 后缀查找因子定义
        lookup = f.replace("_raw", "") if f.endswith("_raw") else f
        fdef = _get_factor(lookup)
        group = fdef.group if fdef else "?"
        print(f"    {f.replace('factor_',''):25s} {w:.3f}  [{group}]")

    return ic_report, selected, final_weights
