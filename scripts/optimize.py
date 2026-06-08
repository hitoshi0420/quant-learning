"""
策略优化三件套

1. 行业偏离约束 — 控制单行业最大权重，避免过度集中
2. 调仓频率对比 — 周/双周/月/季 四种频率回测，权衡换手成本与时效
3. 市场状态择时 — 识别牛/熊/震荡，自适应调整因子权重和仓位

用法:
    python optimize.py                 # 全量优化
    python optimize.py --plot          # 含图表
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import build_clean_factor_table, load_all
from data_fetcher import load_industries
from factor_test import ic_analysis
from multi_factor import select_factors, compute_composite_score


# ============================================================
# 0. 工具函数
# ============================================================

def compute_performance(rets: np.ndarray, periods_per_year: int = 12) -> dict:
    """从收益序列计算绩效指标"""
    mean_ret = np.mean(rets)
    ann_ret = (1 + mean_ret) ** periods_per_year - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(periods_per_year)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum = np.cumprod(1 + rets)
    max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    win_rate = (rets > 0).mean()
    return {
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 3),
        "cum_return": cum[-1],
    }


# ============================================================
# 1. 行业偏离约束
# ============================================================

def backtest_with_industry_constraint(
    factor_table: pd.DataFrame,
    industry_map: pd.DataFrame,
    score_col: str = "composite_score",
    top_n: int = 30,
    max_industry_pct: float = 0.25,
    transaction_cost: float = 0.003,
) -> dict:
    """
    带行业约束的选股回测

    策略: 每月选 Top N，但每个行业不超过 max_industry_pct × N 只
          剩余名额顺延给次优股票
    """
    df = factor_table.copy()
    ind_map = industry_map[["code", "industry"]].drop_duplicates(subset="code")
    df = df.merge(ind_map, on="code", how="left")
    if "industry" not in df.columns:
        df["industry"] = "未知"
    else:
        df["industry"] = df["industry"].fillna("未知")

    prev_portfolio = set()
    monthly_rets = []

    for date in sorted(df["date"].unique()):
        month = df[df["date"] == date].dropna(subset=[score_col]).copy()
        if len(month) < 10:
            continue

        month = month.sort_values(score_col, ascending=False)
        max_per_industry = max(int(top_n * max_industry_pct), 3)

        selected = []
        industry_count = {}
        for _, stock in month.iterrows():
            ind = stock["industry"]
            count = industry_count.get(ind, 0)
            if count < max_per_industry:
                selected.append(stock["code"])
                industry_count[ind] = count + 1
            if len(selected) >= top_n:
                break

        selected_df = month[month["code"].isin(selected)]

        # 换手
        current = set(selected)
        turnover = len(current - prev_portfolio) / len(current) if prev_portfolio else 1.0
        prev_portfolio = current

        gross = selected_df["forward_ret_1m"].mean()
        net = gross - transaction_cost * turnover * 2

        # 行业集中度
        ind_counts = selected_df["industry"].value_counts()
        hhi = (ind_counts / len(selected_df)).pow(2).sum()

        monthly_rets.append({
            "date": date, "return": net, "gross_return": gross,
            "turnover": turnover, "n_selected": len(selected),
            "industry_hhi": hhi,
            "top_industry": ind_counts.index[0],
            "top_ind_pct": ind_counts.iloc[0] / len(selected),
        })

    perf = pd.DataFrame(monthly_rets)
    perf["cum_return"] = (1 + perf["return"]).cumprod()
    stats = compute_performance(perf["return"].values)
    stats["avg_turnover"] = round(perf["turnover"].mean(), 3)
    stats["avg_hhi"] = round(perf["industry_hhi"].mean(), 3)
    stats["max_ind_pct"] = round(perf["top_ind_pct"].max() * 100, 1)
    stats["perf"] = perf

    return stats


# ============================================================
# 2. 调仓频率对比
# ============================================================

def backtest_frequency_scan(
    daily: pd.DataFrame,
    fin: pd.DataFrame,
    industry_map: pd.DataFrame,
    frequencies: list | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    对比不同调仓频率的表现

    频率: weekly, biweekly, monthly, quarterly
    每种频率重构因子表、合成得分、回测
    """
    if frequencies is None:
        frequencies = ["weekly", "biweekly", "monthly", "quarterly"]

    freq_map = {
        "weekly": ("W", 52, "每周"),
        "biweekly": ("2W", 26, "双周"),
        "monthly": ("M", 12, "每月"),
        "quarterly": ("Q", 4, "每季"),
    }

    results = []
    for freq in frequencies:
        print(f"\n{'=' * 50}")
        print(f"测试: {freq}")
        print(f"{'=' * 50}")

        offset, periods_per_year, _ = freq_map[freq]

        # 从日线构建对应频率的因子表
        df = _build_frequency_factor_table(daily, fin, freq=offset)

        # 合成得分
        factor_cols = [c for c in df.columns if c.startswith("factor_")]
        if len(factor_cols) < 3:
            print(f"  {freq}: 因子不足，跳过")
            continue

        # IC 分析选因子
        ic_rpt = ic_analysis(df)
        selected = select_factors(df, ic_rpt, min_ic_ir=0.08, min_t=1.2,
                                  prefer_raw=True)
        if len(selected) < 3:
            top5 = ic_rpt.head(5)["factor"].tolist()
            selected = [f"factor_{f}" if not f.startswith("factor_") else f
                        for f in top5]
            selected = [f for f in selected if f in df.columns]

        scored = compute_composite_score(df, selected, weighting="equal")

        # 确定 top_n
        n_stocks = df.groupby("date")["code"].nunique().mean()
        n_select = min(top_n, int(n_stocks * 0.2))

        # 回测
        prev = set()
        rets = []
        turnovers = []
        for date in sorted(scored["date"].unique()):
            cross = scored[scored["date"] == date].dropna(
                subset=["composite_score", "forward_ret_1m"]
            )
            if len(cross) < 10:
                continue

            cross = cross.sort_values("composite_score", ascending=False)
            selected_stocks = cross.head(n_select)

            current = set(selected_stocks["code"].tolist())
            to = len(current - prev) / len(current) if prev else 1.0
            prev = current

            gross = selected_stocks["forward_ret_1m"].mean()
            # 按年化频率折算交易成本
            cost = 0.003 * to * 2

            rets.append(gross - cost)
            turnovers.append(to)

        if len(rets) < 6:
            print(f"  {freq}: 回测期不足")
            continue

        stats = compute_performance(np.array(rets), periods_per_year)
        stats["frequency"] = freq
        stats["avg_turnover"] = round(np.mean(turnovers), 3)
        stats["cost_drag"] = round(
            np.mean(turnovers) * 0.003 * 2 * periods_per_year * 100, 2
        )
        results.append(stats)

        print(f"  年化: {stats['ann_return']:+.1f}%, "
              f"夏普: {stats['sharpe']:.2f}, "
              f"回撤: {stats['max_dd']:.1f}%, "
              f"换手: {stats['avg_turnover']:.1%}, "
              f"成本拖累: {stats['cost_drag']:.1f}%")

    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


def _build_frequency_factor_table(daily, fin, freq="M"):
    """构建指定频率的因子表"""
    from factors import compute_daily_factors, _align_financials

    df = compute_daily_factors(daily, fin)
    fin_daily = _align_financials(df[["date"]], fin)

    fin_factor_map = {
        "roe_avg": "factor_roe", "np_margin": "factor_np_margin",
        "gp_margin": "factor_gp_margin", "yoy_ni": "factor_yoy_ni",
        "yoy_equity": "factor_yoy_equity",
        "current_ratio": "factor_current_ratio",
        "quick_ratio": "factor_quick_ratio",
        "liability_to_asset": "factor_liability_to_asset",
        "asset_turn_ratio": "factor_asset_turn",
        "cfo_to_np": "factor_cfo_to_np",
    }
    available = {k: v for k, v in fin_factor_map.items() if k in fin_daily.columns}
    for src, dst in available.items():
        fin_daily[dst] = fin_daily[src]
        if src == "liability_to_asset":
            fin_daily[dst] = -fin_daily[src]

    df = df.merge(
        fin_daily[["date", "code"] + list(available.values())],
        on=["date", "code"], how="left",
    )
    for col in available.values():
        df[col] = df.groupby("code")[col].transform(lambda x: x.ffill())

    # 频率重采样
    df["period"] = df["date"].dt.to_period(freq)
    period_dates = df.groupby("period")["date"].max().reset_index(drop=True)
    period_dates = sorted(period_dates.tolist())

    result = df[df["date"].isin(period_dates)].copy()
    result = result.sort_values(["code", "date"])
    result["forward_close"] = result.groupby("code")["close"].shift(-1)
    result["forward_ret_1m"] = result["forward_close"] / result["close"] - 1

    factor_cols = [c for c in result.columns if c.startswith("factor_")]
    result = result[["date", "code"] + factor_cols + ["forward_ret_1m"]].dropna(
        subset=["forward_ret_1m"]
    )
    return result.sort_values(["date", "code"]).reset_index(drop=True)


# ============================================================
# 3. 市场状态择时
# ============================================================

def detect_market_regime(daily: pd.DataFrame) -> pd.DataFrame:
    """
    识别市场状态（基于 HS300 等权指数）

    规则:
      - 牛市: 20日均线 > 60日均线 且 60日均线斜率 > 0
      - 熊市: 20日均线 < 60日均线 且 60日均线斜率 < 0
      - 震荡: 其他

    返回: DataFrame [date, regime, regime_score]
      regime_score: -1(熊) ~ 0(震荡) ~ +1(牛)
    """
    # 构建等权指数
    daily_idx = daily.groupby("date")["close"].mean().reset_index()
    daily_idx = daily_idx.sort_values("date")

    daily_idx["ma_20"] = daily_idx["close"].rolling(20).mean()
    daily_idx["ma_60"] = daily_idx["close"].rolling(60).mean()
    daily_idx["ma_60_slope"] = daily_idx["ma_60"].diff(20) / 20  # 月度斜率

    # 标准化斜率
    slope_mean = daily_idx["ma_60_slope"].rolling(60).mean()
    slope_std = daily_idx["ma_60_slope"].rolling(60).std()
    daily_idx["slope_z"] = (
        (daily_idx["ma_60_slope"] - slope_mean) / slope_std
    ).clip(-3, 3)

    # 趋势强度
    daily_idx["trend_strength"] = (
        (daily_idx["close"] / daily_idx["ma_60"] - 1) * 100
    )

    def classify(row):
        if pd.isna(row["ma_20"]) or pd.isna(row["ma_60"]):
            return "unknown", 0
        above_ma = row["ma_20"] > row["ma_60"]
        slope_pos = row["slope_z"] > 0.3
        slope_neg = row["slope_z"] < -0.3

        if above_ma and slope_pos:
            return "bull", row["slope_z"]
        elif (not above_ma) and slope_neg:
            return "bear", row["slope_z"]
        else:
            return "sideways", 0

    regimes = daily_idx[["date"]].copy()
    regimes[["regime", "score"]] = daily_idx.apply(
        lambda r: pd.Series(classify(r)), axis=1
    )
    regimes["trend_strength"] = daily_idx["trend_strength"]

    return regimes


def backtest_with_timing(
    factor_table: pd.DataFrame,
    regimes: pd.DataFrame,
    score_col: str = "composite_score",
    top_n: int = 30,
    timing_mode: str = "position",  # "position" 仓位 or "factor" 因子轮动
) -> dict:
    """
    市场状态自适应回测

    timing_mode:
      - "position": 牛市满仓, 熊市半仓, 震荡 75% 仓位
      - "factor":   不同市场状态用不同因子（动量→牛市, 质量→熊市, 反转→震荡）
    """
    df = factor_table.merge(regimes[["date", "regime", "score"]], on="date", how="left")
    df["regime"] = df["regime"].fillna("sideways")

    prev_portfolio = set()
    monthly_rets = []

    for date in sorted(df["date"].unique()):
        month = df[df["date"] == date].dropna(subset=[score_col]).copy()
        if len(month) < 10:
            continue

        regime = month["regime"].iloc[0]
        regime_score = month["score"].iloc[0]

        # 仓位调整
        if timing_mode == "position":
            if regime == "bull":
                position = 1.0
            elif regime == "bear":
                position = 0.5
            else:
                position = 0.75
        else:
            position = 1.0

        # 选股
        month = month.sort_values(score_col, ascending=False)
        selected = month.head(int(top_n * position))

        current = set(selected["code"].tolist())
        turnover = len(current - prev_portfolio) / len(current) if prev_portfolio else 1.0
        prev_portfolio = current

        gross = selected["forward_ret_1m"].mean() * position
        net = gross - 0.003 * turnover * 2

        monthly_rets.append({
            "date": date, "return": net, "gross_return": gross,
            "turnover": turnover, "regime": regime,
            "position": position, "n_selected": len(selected),
        })

    perf = pd.DataFrame(monthly_rets)
    if len(perf) == 0:
        return {}

    perf["cum_return"] = (1 + perf["return"]).cumprod()
    stats = compute_performance(perf["return"].values)
    stats["perf"] = perf
    stats["avg_turnover"] = round(perf["turnover"].mean(), 3)

    # 分市场状态统计
    for r in ["bull", "sideways", "bear"]:
        sub = perf[perf["regime"] == r]
        if len(sub) > 0:
            sub_stats = compute_performance(sub["return"].values)
            stats[f"{r}_months"] = len(sub)
            stats[f"{r}_ann_ret"] = sub_stats["ann_return"]
            stats[f"{r}_sharpe"] = sub_stats["sharpe"]

    return stats


# ============================================================
# 4. 主流程：三合一优化报告
# ============================================================

def run_full_optimization(plot: bool = False, top_n: int = 30):
    """运行全部三项优化"""
    print("=" * 60)
    print("策略优化三件套")
    print("=" * 60)

    # 基础数据
    print("\n[加载数据]")
    daily, fin, industry = load_all(ROOT)
    factor_table = build_clean_factor_table(ROOT, start_date="2021-01-01",
                                            neutralize=False)

    # 因子筛选 + 得分
    print("\n[因子筛选 & 合成]")
    ic_report = ic_analysis(factor_table)
    factor_cols = select_factors(factor_table, ic_report,
                                 min_ic_ir=0.10, min_t=1.5, prefer_raw=True)
    scored = compute_composite_score(factor_table, factor_cols, weighting="equal")

    # ---- 优化 1: 行业约束 ----
    print("\n" + "=" * 60)
    print("优化 1/3: 行业偏离约束")
    print("=" * 60)

    constraints = [0.20, 0.25, 0.30, 1.0]  # 1.0 = 无约束
    ind_results = []
    for cap in constraints:
        label = f"单行业≤{int(cap*100)}%" if cap < 1 else "无约束"
        bt = backtest_with_industry_constraint(
            scored, industry, top_n=top_n, max_industry_pct=cap
        )
        bt["constraint"] = label
        ind_results.append(bt)
        print(f"  {label:16s}: 年化 {bt['ann_return']:+.1f}%, "
              f"夏普 {bt['sharpe']:.2f}, 回撤 {bt['max_dd']:.1f}%, "
              f"最大行业 {bt['max_ind_pct']:.1f}%")

    # ---- 优化 2: 调仓频率 ----
    print("\n" + "=" * 60)
    print("优化 2/3: 调仓频率对比")
    print("=" * 60)

    freq_results = backtest_frequency_scan(daily, fin, industry, top_n=top_n)

    # ---- 优化 3: 市场择时 ----
    print("\n" + "=" * 60)
    print("优化 3/3: 市场状态择时")
    print("=" * 60)

    print("识别市场状态...")
    regimes = detect_market_regime(daily)
    regime_counts = regimes["regime"].value_counts()
    print(f"  市场状态分布: {dict(regime_counts)}")

    # 无择时基准
    print("\n无择时基准:")
    no_timing = backtest_with_timing(scored, regimes, top_n=top_n,
                                     timing_mode="position")
    # 直接用默认仓位1.0的回测作为基准
    from multi_factor import backtest_portfolio
    baseline = backtest_portfolio(scored, top_n=top_n)
    print(f"  年化: {baseline['ann_return']:+.1f}%, "
          f"夏普: {baseline['sharpe']:.2f}, 回撤: {baseline['max_dd']:.1f}%")

    # 仓位择时
    print("\n仓位择时 (牛1.0/震0.75/熊0.5):")
    timed = backtest_with_timing(scored, regimes, top_n=top_n,
                                 timing_mode="position")
    if timed:
        print(f"  年化: {timed['ann_return']:+.1f}%, "
              f"夏普: {timed['sharpe']:.2f}, 回撤: {timed['max_dd']:.1f}%")
        for r in ["bull", "sideways", "bear"]:
            if f"{r}_months" in timed:
                print(f"    {r}: {timed[f'{r}_months']} 个月, "
                      f"年化 {timed[f'{r}_ann_ret']:+.1f}%, "
                      f"夏普 {timed[f'{r}_sharpe']:.2f}")

    # ---- 综合对比 ----
    print("\n" + "=" * 60)
    print("综合对比")
    print("=" * 60)

    summary_rows = [
        {"优化": "原始策略", "年化收益": baseline["ann_return"],
         "夏普": baseline["sharpe"], "最大回撤": baseline["max_dd"],
         "月均换手": round(baseline["avg_turnover"] * 100)},
    ]

    best_ind = min(ind_results, key=lambda x: (-x["sharpe"], x["max_dd"]))
    summary_rows.append({
        "优化": f"行业约束({best_ind['constraint']})",
        "年化收益": best_ind["ann_return"],
        "夏普": best_ind["sharpe"],
        "最大回撤": best_ind["max_dd"],
        "月均换手": round(best_ind["avg_turnover"] * 100),
    })

    best_freq = freq_results.iloc[0] if len(freq_results) > 0 else None
    if best_freq is not None:
        summary_rows.append({
            "优化": f"调仓频率({best_freq['frequency']})",
            "年化收益": best_freq["ann_return"],
            "夏普": best_freq["sharpe"],
            "最大回撤": best_freq["max_dd"],
            "月均换手": round(best_freq["avg_turnover"] * 100),
        })

    if timed:
        summary_rows.append({
            "优化": "市场择时(仓位)",
            "年化收益": timed["ann_return"],
            "夏普": timed["sharpe"],
            "最大回撤": timed["max_dd"],
            "月均换手": round(timed["avg_turnover"] * 100),
        })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    # 保存
    pd.DataFrame(ind_results).to_csv(
        ROOT / "data" / "optimize_industry.csv", index=False, encoding="utf-8-sig")
    freq_results.to_csv(
        ROOT / "data" / "optimize_frequency.csv", index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: data/optimize_*.csv")

    # 图表
    if plot:
        plot_optimization_summary(baseline, timed, ind_results, freq_results,
                                  ROOT / "data" / "plots")

    return summary


def plot_optimization_summary(baseline, timed, ind_results, freq_results, save_dir):
    """优化对比全景图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 12))

    # --- 图1: 行业约束 vs 无约束净值 ---
    ax1 = fig.add_subplot(2, 3, 1)
    colors = ["#e53935", "#ff9800", "#2196f3", "#4caf50"]
    for i, r in enumerate(ind_results):
        if "perf" in r:
            p = r["perf"]
            ax1.plot(p["date"], p["cum_return"], color=colors[i],
                     linewidth=1.5, label=f"{r['constraint']} (夏普={r['sharpe']:.2f})")
    ax1.set_title("行业偏离约束效果", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # --- 图2: 调仓频率对比 ---
    ax2 = fig.add_subplot(2, 3, 2)
    if len(freq_results) > 0:
        freqs = freq_results["frequency"].tolist()
        sharpes = freq_results["sharpe"].tolist()
        returns = freq_results["ann_return"].tolist()
        x = range(len(freqs))
        ax2.bar([i - 0.15 for i in x], returns, 0.3, color="#e53935",
                label="年化收益%")
        ax2_twin = ax2.twinx()
        ax2_twin.bar([i + 0.15 for i in x], sharpes, 0.3, color="#2196f3",
                      label="夏普")
        ax2.set_xticks(x)
        ax2.set_xticklabels(freqs)
        ax2.set_title("调仓频率对比", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=8)
        ax2_twin.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.2)

    # --- 图3: 市场状态净值(择时 vs 不择时) ---
    ax3 = fig.add_subplot(2, 3, 3)
    if "perf" in baseline:
        bp = baseline["perf"]
        ax3.plot(bp["date"], bp["cum_return"], color="#607d8b",
                 linewidth=1.5, linestyle="--", label="无择时")
    if timed and "perf" in timed:
        tp = timed["perf"]
        ax3.plot(tp["date"], tp["cum_return"], color="#e53935",
                 linewidth=1.5, label="仓位择时")
    ax3.set_title("市场择时 vs 不择时", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # --- 图4: 市场状态分布 ---
    ax4 = fig.add_subplot(2, 3, 4)
    if timed and "perf" in timed:
        tp = timed["perf"]
        for r, color, label in [("bull", "#e53935", "牛市"),
                                 ("sideways", "#ff9800", "震荡"),
                                 ("bear", "#4caf50", "熊市")]:
            sub = tp[tp["regime"] == r]
            if len(sub) > 0:
                # 画净值
                sub = sub.copy()
                sub["seg_cum"] = (1 + sub["return"]).cumprod()
                ax4.plot(range(len(sub)), sub["seg_cum"].values,
                         color=color, linewidth=1.5, label=f"{label}({len(sub)}月)")
    ax4.set_title("各市场状态净值", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    # --- 图5: 优化效果汇总 ---
    ax5 = fig.add_subplot(2, 3, 5)
    labels = ["原始"]
    sharpes = [baseline.get("sharpe", 0)]
    returns = [baseline.get("ann_return", 0)]

    best_ind = min(ind_results, key=lambda x: (-x.get("sharpe", 0), x.get("max_dd", 99)))
    labels.append("行业约束")
    sharpes.append(best_ind.get("sharpe", 0))
    returns.append(best_ind.get("ann_return", 0))

    if len(freq_results) > 0:
        labels.append(freq_results.iloc[0]["frequency"])
        sharpes.append(freq_results.iloc[0]["sharpe"])
        returns.append(freq_results.iloc[0]["ann_return"])

    if timed:
        labels.append("择时")
        sharpes.append(timed.get("sharpe", 0))
        returns.append(timed.get("ann_return", 0))

    x = range(len(labels))
    ax5.bar([i - 0.15 for i in x], returns, 0.3, color="#e53935", label="年化收益%")
    ax5_twin = ax5.twinx()
    ax5_twin.bar([i + 0.15 for i in x], sharpes, 0.3, color="#2196f3", label="夏普")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_title("优化效果汇总", fontsize=12, fontweight="bold")
    ax5.legend(loc="upper left", fontsize=8)
    ax5_twin.legend(loc="upper right", fontsize=8)

    # --- 图6: 优化参数表 ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    table_data = [
        ["原始策略", f"{baseline.get('ann_return',0):+.1f}%",
         f"{baseline.get('sharpe',0):.2f}", f"{baseline.get('max_dd',0):.1f}%"],
        ["行业约束", f"{best_ind.get('ann_return',0):+.1f}%",
         f"{best_ind.get('sharpe',0):.2f}", f"{best_ind.get('max_dd',0):.1f}%"],
    ]
    if len(freq_results) > 0:
        table_data.append([
            f"频率-{freq_results.iloc[0]['frequency']}",
            f"{freq_results.iloc[0]['ann_return']:+.1f}%",
            f"{freq_results.iloc[0]['sharpe']:.2f}",
            f"{freq_results.iloc[0]['max_dd']:.1f}%",
        ])
    if timed:
        table_data.append([
            "市场择时",
            f"{timed.get('ann_return',0):+.1f}%",
            f"{timed.get('sharpe',0):.2f}",
            f"{timed.get('max_dd',0):.1f}%",
        ])

    table = ax6.table(
        cellText=table_data,
        colLabels=["优化方案", "年化收益", "夏普", "最大回撤"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for j in range(4):
        table[0, j].set_facecolor("#e53935")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax6.set_title("综合对比", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "optimization_summary.png", dpi=150, bbox_inches="tight")
    print(f"图表已保存: {save_dir / 'optimization_summary.png'}")
    plt.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="策略优化")
    parser.add_argument("--plot", action="store_true", help="输出图表")
    parser.add_argument("--top", type=int, default=30, help="持仓数")
    args = parser.parse_args()

    run_full_optimization(plot=args.plot, top_n=args.top)
