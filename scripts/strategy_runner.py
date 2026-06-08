"""
多策略回测框架 — 策略配置 + 并行回测 + 对比输出
"""

import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


# ============================================================
# 0. 策略配置
# ============================================================

@dataclass
class StrategyConfig:
    name: str
    factor_groups: list[str]          # 使用的因子分组
    weighting: str = "ic_ir"          # equal / ic_ir / max_sharpe
    universe: str = "top800"          # top800 / full
    liquidity_top_n: int = 800
    top_n: int = 30                   # 持仓数
    ic_lookback: int = 36
    ic_decay: int = 12
    min_ic_ir: float = 0.08
    decorrelate: bool = True
    decorr_threshold: float = 0.7
    neutralize: bool = False
    max_group_weight: dict = None
    description: str = ""


# ============================================================
# 1. 回测引擎
# ============================================================

def run_single_backtest(factor_table: pd.DataFrame,
                        config: StrategyConfig,
                        prices: dict,
                        industry_map: dict,
                        name_map: dict,
                        initial_capital: float = 1.0,
                        commission: float = 0.0003,
                        stamp: float = 0.0005,
                        slippage: float = 0.001) -> pd.DataFrame:
    """
    单策略回测，返回月度收益序列
    """
    from factor_library import FACTOR_GROUPS, get_factor
    from ic_estimator import run_ic_pipeline

    # 筛选可用因子
    group_factors = []
    for g in config.factor_groups:
        if g in FACTOR_GROUPS:
            group_factors.extend(FACTOR_GROUPS[g])
    factor_cols = [c for c in group_factors
                   if c in factor_table.columns and c != "factor_size"]

    if len(factor_cols) < 3:
        print(f"  {config.name}: 可用因子不足 ({len(factor_cols)}), 跳过")
        return pd.DataFrame()

    # Walk-forward IC估计（避免前视偏差）
    # 前24个月作为初始训练，之后每12个月重新估计一次因子权重
    print(f"\n── {config.name} ──")
    filtered_cols = ["date", "code", "forward_ret_1m"] + factor_cols
    filtered_table = factor_table[filtered_cols].copy()

    from config import IC_LOOKBACK_MONTHS, REFIT_EVERY_MONTHS
    all_dates = sorted(factor_table["date"].unique())
    rebalance_dates = all_dates[:-1]  # 最后一个月需要forward_ret
    min_train_months = IC_LOOKBACK_MONTHS  # 从 config 读取
    refit_every = REFIT_EVERY_MONTHS

    selected_factors = factor_cols.copy()
    factor_weights = {f: 1.0 / len(factor_cols) for f in factor_cols}  # 初始等权
    ic_signs = {f: 1 for f in factor_cols}
    last_refit_date = None

    monthly_returns = []
    for i, date in enumerate(tqdm(rebalance_dates, desc=config.name, leave=False)):
        # Walk-forward IC重估计：只用当前日期之前的数据
        train_data = filtered_table[filtered_table["date"] <= date]
        n_train_months = train_data["date"].nunique()

        should_refit = (
            n_train_months >= min_train_months and
            (last_refit_date is None or
             len([d for d in rebalance_dates if last_refit_date < d <= date]) >= refit_every)
        )

        if should_refit:
            try:
                ic_report, selected_factors, factor_weights = run_ic_pipeline(
                    train_data, min_ic_ir=config.min_ic_ir,
                    correlation_threshold=config.decorr_threshold if config.decorrelate else 1.0,
                    max_group_weight=config.max_group_weight,
                )
                ic_signs = {}
                for _, row in ic_report.iterrows():
                    ic_signs[row["factor"]] = 1 if row["ic_mean"] >= 0 else -1
                last_refit_date = date
            except Exception as e:
                print(f"  [{config.name}] IC重估计失败 ({date.date()}): {e}，沿用上次权重")

        # 当日截面
        month = factor_table[factor_table["date"] == date].copy()
        if len(month) < config.top_n:
            continue

        # 流动性筛选
        if "factor_size" in month.columns:
            month = month.sort_values("factor_size", ascending=True).head(config.liquidity_top_n)

        # 打分
        month["score"] = 0.0
        weight_sum = 0.0
        for f in selected_factors:
            if f in month.columns:
                sign = ic_signs.get(f, 1)
                w = factor_weights.get(f, 0)
                month["score"] += month[f].fillna(0) * sign * w
                weight_sum += abs(w)
        if weight_sum > 0:
            month["score"] /= weight_sum

        # 选 top N
        top = month.sort_values("score", ascending=False).head(config.top_n)
        if len(top) < 5:
            continue

        # 等权组合收益 = mean(forward_ret_1m)
        ret = top["forward_ret_1m"].mean()

        # 扣除交易成本
        ret -= (commission * 2 + stamp + slippage * 2)

        monthly_returns.append({"date": date, "return": ret, "strategy": config.name})

    if not monthly_returns:
        return pd.DataFrame()

    return pd.DataFrame(monthly_returns)


# ============================================================
# 2. 基准
# ============================================================

def compute_benchmark(factor_table: pd.DataFrame,
                      liquidity_top_n: int = 800) -> pd.DataFrame:
    """TopN 等权基准收益"""
    all_dates = sorted(factor_table["date"].unique())
    rebalance_dates = all_dates[:-1]

    bench_returns = []
    for date in rebalance_dates:
        month = factor_table[factor_table["date"] == date].copy()
        if len(month) < 10:
            continue
        if "factor_size" in month.columns:
            month = month.sort_values("factor_size", ascending=True).head(liquidity_top_n)
        ret = month["forward_ret_1m"].mean()
        bench_returns.append({"date": date, "return": ret, "strategy": "Benchmark"})

    return pd.DataFrame(bench_returns)


def compute_index_proxies(factor_table: pd.DataFrame) -> pd.DataFrame:
    """
    用市值排名构建 CSI 300 / CSI 500 / CSI 800 代理指数

    - CSI800: factor_size 最大的 800 只（对应全市场大中盘）
    - CSI300: factor_size 最大的 300 只（大盘蓝筹）
    - CSI500: factor_size 第 301-800 名（中盘）

    返回: DataFrame [date, return, strategy] 包含三个基准的月度收益
    """
    all_dates = sorted(factor_table["date"].unique())
    rebalance_dates = all_dates[:-1]

    if "factor_size" not in factor_table.columns:
        return pd.DataFrame()

    records = []
    for date in rebalance_dates:
        month = factor_table[factor_table["date"] == date].copy()
        if len(month) < 800:
            continue

        ranked = month.sort_values("factor_size", ascending=False)

        csi800 = ranked.head(800)
        csi300 = ranked.head(300)
        csi500 = ranked.iloc[300:800] if len(ranked) >= 800 else ranked.iloc[300:]

        for name, subset in [("CSI800", csi800), ("CSI300", csi300), ("CSI500", csi500)]:
            if len(subset) < 10:
                continue
            ret = subset["forward_ret_1m"].mean()
            records.append({"date": date, "return": ret, "strategy": name})

    return pd.DataFrame(records)


# ============================================================
# 3. 多策略运行
# ============================================================

def run_strategy_comparison(factor_table: pd.DataFrame,
                            configs: list[StrategyConfig],
                            prices: dict = None,
                            industry_map: dict = None,
                            name_map: dict = None) -> pd.DataFrame:
    """
    运行多策略回测对比

    返回: 绩效对比 DataFrame
    """
    # 基准
    print("计算基准...")
    bench = compute_benchmark(factor_table)
    csi_proxies = compute_index_proxies(factor_table)

    # 运行各策略
    all_returns = [bench]
    if len(csi_proxies) > 0:
        all_returns.append(csi_proxies)
    for config in configs:
        if config.name == "S7_多策略集成":
            rets = run_ensemble_backtest(factor_table)
        else:
            rets = run_single_backtest(factor_table, config,
                                       prices or {}, industry_map or {},
                                       name_map or {})
        if len(rets) > 0:
            all_returns.append(rets)

    if not all_returns:
        return pd.DataFrame()

    returns_df = pd.concat(all_returns, ignore_index=True)

    # 计算绩效指标
    perf = []
    # 基准收益（用于计算信息比率）
    primary_bench_rets = None
    all_bench_rets = {}
    for bench_name in ["CSI800", "CSI300", "CSI500", "Benchmark"]:
        if bench_name in returns_df["strategy"].values:
            b_rets = returns_df[returns_df["strategy"] == bench_name]["return"]
            all_bench_rets[bench_name] = b_rets
            if primary_bench_rets is None:
                primary_bench_rets = b_rets

    for name in returns_df["strategy"].unique():
        strat_rets = returns_df[returns_df["strategy"] == name]["return"]
        if len(strat_rets) < 12:
            continue

        ann_ret = strat_rets.mean() * 12
        ann_vol = strat_rets.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # 最大回撤 + 平均回撤持续时间
        cum = (1 + strat_rets).cumprod()
        peak = cum.expanding().max()
        dd = (cum / peak - 1)
        max_dd = dd.min()
        # 平均回撤持续时间（月）
        in_dd = dd < -0.01
        dd_durations = []
        current = 0
        for flag in in_dd:
            if flag:
                current += 1
            else:
                if current > 0:
                    dd_durations.append(current)
                    current = 0
        if current > 0:
            dd_durations.append(current)
        avg_dd_months = np.mean(dd_durations) if dd_durations else 0

        # 最大连续亏损月数
        losing = strat_rets.values < 0
        max_consec_loss = 0
        current_loss = 0
        for flag in losing:
            if flag:
                current_loss += 1
                max_consec_loss = max(max_consec_loss, current_loss)
            else:
                current_loss = 0

        # 胜率
        win_rate = (strat_rets > 0).mean()

        # Calmar
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0

        # 累计收益
        cum_ret = cum.iloc[-1] - 1

        # Sortino 比率（下行波动率）
        down_rets = strat_rets[strat_rets < 0]
        down_vol = down_rets.std() * np.sqrt(12) if len(down_rets) > 0 else 0
        sortino = ann_ret / down_vol if down_vol > 0 else 0

        # 信息比率（相对 CSI800 主基准）
        info_ratio = 0
        ir_vs_300 = 0
        ir_vs_500 = 0
        if primary_bench_rets is not None and not name.startswith("CSI") and name != "Benchmark":
            excess = strat_rets.values - primary_bench_rets.values
            excess_mean = excess.mean() * 12
            excess_vol = excess.std() * np.sqrt(12)
            info_ratio = excess_mean / excess_vol if excess_vol > 0 else 0

            # IR vs CSI300
            if "CSI300" in all_bench_rets:
                exc_300 = strat_rets.values - all_bench_rets["CSI300"].values
                em = exc_300.mean() * 12
                ev = exc_300.std() * np.sqrt(12)
                ir_vs_300 = em / ev if ev > 0 else 0

            # IR vs CSI500
            if "CSI500" in all_bench_rets:
                exc_500 = strat_rets.values - all_bench_rets["CSI500"].values
                em = exc_500.mean() * 12
                ev = exc_500.std() * np.sqrt(12)
                ir_vs_500 = em / ev if ev > 0 else 0

        # 滚动12月夏普稳定性
        rolling_sharpes = []
        if len(strat_rets) >= 24:
            for i in range(12, len(strat_rets) + 1):
                window = strat_rets.iloc[max(0, i-12):i]
                if len(window) >= 6:
                    w_ann = window.mean() * 12
                    w_vol = window.std() * np.sqrt(12)
                    rolling_sharpes.append(w_ann / w_vol if w_vol > 0 else 0)
        rolling_sharpe_std = np.std(rolling_sharpes) if rolling_sharpes else 0

        perf.append({
            "strategy": name,
            "ann_return": ann_ret,
            "ann_volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "win_rate": win_rate,
            "cum_return": cum_ret,
            "info_ratio": info_ratio,
            "ir_vs_csi300": round(ir_vs_300, 3),
            "ir_vs_csi500": round(ir_vs_500, 3),
            "max_consec_loss": max_consec_loss,
            "avg_dd_months": round(avg_dd_months, 1),
            "rolling_sharpe_std": round(rolling_sharpe_std, 3),
            "n_months": len(strat_rets),
        })

    return pd.DataFrame(perf).sort_values("sharpe", ascending=False)


# ============================================================
# 4. 多策略集成回测
# ============================================================

from factor_library import STRATEGY_CLUSTERS


def run_ensemble_backtest(factor_table: pd.DataFrame,
                          total_picks: int = 10,
                          liquidity_top_n: int = 800,
                          commission: float = 0.0003,
                          stamp: float = 0.0005,
                          slippage: float = 0.001) -> pd.DataFrame:
    """
    多策略集成回测：每个簇独立IC分析（一次）+ 每月打分合并

    1. IC分析：每簇在整个历史上运行一次 → 确定因子+权重+方向
    2. 每月：每簇独立打分 → 取top-N → 合并去重 → 计算收益
    """
    from factor_library import FACTOR_GROUPS
    from ic_estimator import run_ic_pipeline

    all_dates = sorted(factor_table["date"].unique())
    rebalance_dates = all_dates[:-1]

    # ── Walk-forward: 每12个月重新估计各簇IC权重 ──
    from config import IC_LOOKBACK_MONTHS, REFIT_EVERY_MONTHS
    min_train_months = IC_LOOKBACK_MONTHS
    refit_every = REFIT_EVERY_MONTHS

    # 预计算各簇的因子列表
    cluster_factor_lists = {}
    for cluster_name, config in STRATEGY_CLUSTERS.items():
        group_factors = []
        for g in config["factor_groups"]:
            if g in FACTOR_GROUPS:
                group_factors.extend(FACTOR_GROUPS[g])
        cluster_factor_cols = [c for c in group_factors
                               if c in factor_table.columns and c != "factor_size"]
        if len(cluster_factor_cols) >= 2:
            cluster_factor_lists[cluster_name] = cluster_factor_cols

    # 初始等权模型
    cluster_models = {}
    for cluster_name, factor_cols in cluster_factor_lists.items():
        cluster_models[cluster_name] = {
            "selected_factors": factor_cols,
            "factor_weights": {f: 1.0 / len(factor_cols) for f in factor_cols},
            "ic_signs": {f: 1 for f in factor_cols},
            "pick_count": STRATEGY_CLUSTERS[cluster_name]["pick_count"],
        }
    last_refit_date = None

    if not cluster_models:
        return pd.DataFrame()

    # ── 逐月回测（Walk-forward IC）──
    monthly_returns = []
    for date in tqdm(rebalance_dates, desc="集成回测", leave=False):
        # Walk-forward IC重估计
        n_train_months = len([d for d in all_dates if d <= date])
        should_refit = (
            n_train_months >= min_train_months and
            (last_refit_date is None or
             len([d for d in rebalance_dates if last_refit_date < d <= date]) >= refit_every)
        )

        if should_refit:
            for cluster_name, factor_cols in cluster_factor_lists.items():
                filtered_cols = ["date", "code", "forward_ret_1m"] + factor_cols
                train_data = factor_table[filtered_cols].copy()
                train_data = train_data[train_data["date"] <= date]
                if train_data["date"].nunique() < min_train_months:
                    continue
                try:
                    ic_report, selected, weights = run_ic_pipeline(
                        train_data, min_ic_ir=0.06, max_group_weight=None,
                    )
                    ic_signs = {}
                    if isinstance(ic_report, pd.DataFrame) and len(ic_report) > 0:
                        for _, row in ic_report.iterrows():
                            ic_signs[row["factor"]] = 1 if row["ic_mean"] >= 0 else -1
                    cluster_models[cluster_name] = {
                        "selected_factors": selected,
                        "factor_weights": weights,
                        "ic_signs": ic_signs,
                        "pick_count": STRATEGY_CLUSTERS[cluster_name]["pick_count"],
                    }
                except Exception as e:
                    print(f"  [集成回测] {cluster_name} IC重估计失败 ({date.date()}): {e}")
            last_refit_date = date
        month = factor_table[factor_table["date"] == date].copy()
        if len(month) < 30:
            continue

        if "factor_size" in month.columns:
            month = month.sort_values("factor_size", ascending=True).head(liquidity_top_n)

        all_picked_codes = set()
        cluster_picks = {}

        for cluster_name, model in cluster_models.items():
            month_cluster = month.copy()
            month_cluster["score"] = 0.0
            weight_sum = 0.0
            for f in model["selected_factors"]:
                if f in month_cluster.columns:
                    sign = model["ic_signs"].get(f, 1)
                    w = model["factor_weights"].get(f, 0)
                    month_cluster["score"] += month_cluster[f].fillna(0) * sign * w
                    weight_sum += abs(w)
            if weight_sum > 0:
                month_cluster["score"] /= weight_sum

            n_picks = model["pick_count"]
            top = month_cluster.sort_values("score", ascending=False).head(n_picks)
            cluster_picks[cluster_name] = top["code"].tolist()
            all_picked_codes.update(top["code"].tolist())

        # 去重后补足
        if len(all_picked_codes) < total_picks:
            for codes in cluster_picks.values():
                for c in codes:
                    if len(all_picked_codes) >= total_picks:
                        break
                    all_picked_codes.add(c)

        picked = month[month["code"].isin(all_picked_codes)]
        if len(picked) < 3:
            continue

        ret = picked["forward_ret_1m"].mean()
        ret -= (commission * 2 + stamp + slippage * 2)

        monthly_returns.append({
            "date": date,
            "return": ret,
            "strategy": "S7_多策略集成",
        })

    if not monthly_returns:
        return pd.DataFrame()
    return pd.DataFrame(monthly_returns)


# ============================================================
# 5. 预设策略
# ============================================================

def get_preset_strategies() -> list[StrategyConfig]:
    """7个内置策略配置"""
    return [
        StrategyConfig(
            name="S1_纯价值",
            factor_groups=["value"],
            description="BP+SP+EP+CFP，深度价值",
        ),
        StrategyConfig(
            name="S2_纯动量",
            factor_groups=["momentum", "reversal"],
            description="动量+反转因子",
        ),
        StrategyConfig(
            name="S3_纯质量",
            factor_groups=["profitability", "quality"],
            description="ROE+利润率+现金流等",
        ),
        StrategyConfig(
            name="S4_全因子等权",
            factor_groups=["value", "momentum", "volatility", "liquidity",
                          "profitability", "growth", "quality", "reversal"],
            weighting="equal",
            decorrelate=False,
            description="所有8组等权",
        ),
        StrategyConfig(
            name="S5_全因子ICIR",
            factor_groups=["value", "momentum", "volatility", "liquidity",
                          "profitability", "growth", "quality", "reversal"],
            weighting="ic_ir",
            decorrelate=True,
            description="全因子动态IC_IR+去相关",
        ),
        StrategyConfig(
            name="S6_去BP化",
            factor_groups=["momentum", "volatility", "liquidity",
                          "profitability", "growth", "quality", "reversal",
                          "value"],  # value放最后，限制权重
            weighting="ic_ir",
            decorrelate=True,
            max_group_weight={"value": 0.15, "momentum": 0.25, "quality": 0.25},
            description="限制价值组权重≤15%，鼓励多样性",
        ),
        StrategyConfig(
            name="S7_多策略集成",
            factor_groups=["value", "momentum", "volatility", "liquidity",
                          "profitability", "growth", "quality", "reversal"],
            weighting="ic_ir",
            decorrelate=True,
            description="三簇并行: 价值+成长+质量，各取topN合并",
        ),
    ]


# ============================================================
# 6. 输出格式化
# ============================================================

def print_comparison(perf_df: pd.DataFrame, benchmark_name: str = "Benchmark"):
    """打印策略对比表格"""
    if len(perf_df) == 0:
        print("无有效回测结果")
        return

    # 找到基准行
    bench_row = perf_df[perf_df["strategy"] == benchmark_name]
    bench_sharpe = bench_row["sharpe"].values[0] if len(bench_row) > 0 else 0

    print(f"\n{'=' * 115}")
    print(f"  策略对比结果 (基准: CSI800 代理)")
    print(f"{'=' * 115}")
    print(f"  {'策略':20s} {'年化':>7s} {'波动':>7s} {'夏普':>6s} "
          f"{'Sortino':>7s} {'IR':>6s} {'回撤':>7s} {'Calmar':>7s} "
          f"{'胜率':>6s} {'连亏':>5s} {'DD月':>5s}")
    print(f"  {'-' * 110}")

    for _, row in perf_df.iterrows():
        print(f"  {row['strategy']:20s} {row['ann_return']:>+6.1%} "
              f"{row['ann_volatility']:>6.1%} {row['sharpe']:>5.2f} "
              f"{row['sortino']:>6.2f} {row['info_ratio']:>5.2f} "
              f"{row['max_drawdown']:>6.1%} {row['calmar']:>6.2f} "
              f"{row['win_rate']:>5.1%} {row['max_consec_loss']:>4.0f} "
              f"{row['avg_dd_months']:>4.1f}")

    # 补充展示 CSI 基准行
    csi_rows = perf_df[perf_df["strategy"].isin(["CSI800", "CSI300", "CSI500"])]
    if len(csi_rows) > 0:
        print(f"\n  {'─' * 110}")
        print(f"  基准指数代理 (按 factor_size 市值排名构造)")
        print(f"  CSI800 = Top800,  CSI300 = Top300 (大盘),  CSI500 = 301-800 (中盘)")
        for _, row in csi_rows.iterrows():
            print(f"  {row['strategy']:20s} {row['ann_return']:>+6.1%} "
                  f"{row['ann_volatility']:>6.1%} {row['sharpe']:>5.2f} "
                  f"{row['sortino']:>6.2f} {'--':>6s} "
                  f"{row['max_drawdown']:>6.1%} {row['calmar']:>6.2f} "
                  f"{row['win_rate']:>5.1%} {row['max_consec_loss']:>4.0f} "
                  f"{row['avg_dd_months']:>4.1f}")

    # IR vs 多基准对比
    non_bench = perf_df[~perf_df["strategy"].str.startswith("CSI")]
    non_bench = non_bench[non_bench["strategy"] != "Benchmark"]
    if len(non_bench) > 0 and "ir_vs_csi300" in non_bench.columns:
        print(f"\n  {'─' * 110}")
        print(f"  信息比率 vs 多基准")
        print(f"  {'策略':20s} {'IR vs CSI800':>12s} {'IR vs CSI300':>12s} {'IR vs CSI500':>12s}")
        for _, row in non_bench.iterrows():
            print(f"  {row['strategy']:20s} {row['info_ratio']:>+11.3f} "
                  f"{row['ir_vs_csi300']:>+11.3f} {row['ir_vs_csi500']:>+11.3f}")

    print(f"  {'-' * 110}")
    print(f"  注: IR=信息比率(超额/跟踪误差 vs CSI800), DD月=平均回撤持续月数, 连亏=最大连续亏损月数")
