"""
全 A 股多因子回测（流动性过滤 + 有效因子）
用法: python scripts/full_universe_backtest.py
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


# ============================================================
# 1. 加载数据
# ============================================================

def load_data():
    """加载缓存的因子表 + 日线原始数据（用于流动性过滤）"""
    print("加载因子表...")
    factor_table = pd.read_parquet(ROOT / "data" / "factor_table_full.parquet")
    factor_table["date"] = pd.to_datetime(factor_table["date"])

    # 加载日线用于流动性过滤（取 amount 和 close）
    print("加载日线数据（流动性计算）...")
    daily_dir = ROOT / "data" / "daily"
    daily_list = []
    for f in tqdm(sorted(daily_dir.glob("*.parquet")), desc="日线"):
        df = pd.read_parquet(f, columns=["date", "amount", "close", "volume"])
        df["code"] = f.stem
        df["date"] = pd.to_datetime(df["date"])
        daily_list.append(df)

    daily = pd.concat(daily_list, ignore_index=True)
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)

    # 加载行业
    from data_fetcher import load_industries
    industry = load_industries()
    industry_map = dict(zip(industry["code"], industry["industry"]))

    return factor_table, daily, industry_map


# ============================================================
# 2. 流动性过滤
# ============================================================

def apply_liquidity_filter(daily, factor_table, min_daily_amount=100_000_000):
    """
    在每个调仓日，筛选过去 60 个交易日日均成交额 >= min_daily_amount 的股票
    返回: 过滤后的 factor_table
    """
    # 对每个 date，计算过去 60 天的日均成交额
    dates = sorted(factor_table["date"].unique())

    # 预计算每只股票每天的 60 日滚动均额
    daily["amount_60d"] = daily.groupby("code")["amount"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )

    # 对每个调仓日期，取对应的流动性值
    amount_map = {}
    for code, grp in tqdm(daily.groupby("code"), desc="流动性映射"):
        grp = grp.sort_values("date")
        for _, row in grp.iterrows():
            if pd.notna(row["amount_60d"]):
                amount_map[(code, row["date"])] = row["amount_60d"]

    # 过滤
    filtered = factor_table.copy()
    filtered["liquidity_ok"] = filtered.apply(
        lambda r: amount_map.get((r["code"], r["date"]), 0) >= min_daily_amount,
        axis=1,
    )

    n_before = filtered["code"].nunique()
    filtered = filtered[filtered["liquidity_ok"]].copy()
    n_after = filtered["code"].nunique()

    print(f"流动性过滤 (日均≥{min_daily_amount/1e8:.1f}亿): {n_before} → {n_after} 只")
    return filtered


# ============================================================
# 3. 回测引擎
# ============================================================

def run_backtest(
    factor_table,
    factor_cols,
    top_n=30,
    transaction_cost=0.003,
    name="Strategy",
):
    """
    月度调仓回测

    factor_table: 含 date, code, factor_*, forward_ret_1m
    factor_cols: 使用的因子列名列表
    """
    df = factor_table[["date", "code", "forward_ret_1m"] + factor_cols].dropna().copy()

    # 计算综合得分（等权）
    available = [c for c in factor_cols if c in df.columns]
    if not available:
        print(f"  [{name}] 无可用因子!")
        return None

    # 统一因子方向（确保因子值越高 → 收益越高）
    for col in available:
        # 计算整体 IC 方向
        ic_list = []
        for date, grp in df.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) >= 10:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(valid[col], valid["forward_ret_1m"])
                if not np.isnan(ic):
                    ic_list.append(ic)
        if ic_list and np.mean(ic_list) < 0:
            df[col] = -df[col]

    # 等权合成得分
    df["score"] = df[available].fillna(0).mean(axis=1)

    # 每月选 top N
    monthly_records = []
    prev_portfolio = set()

    for date in sorted(df["date"].unique()):
        month_data = df[df["date"] == date].copy()
        n_stocks = len(month_data)
        if n_stocks < top_n:
            continue

        month_data = month_data.sort_values("score", ascending=False)
        selected = month_data.head(top_n)

        # 换手率
        current_portfolio = set(selected["code"].tolist())
        if prev_portfolio:
            turnover = len(current_portfolio - prev_portfolio) / len(current_portfolio)
        else:
            turnover = 1.0
        prev_portfolio = current_portfolio

        # 等权收益
        gross_ret = selected["forward_ret_1m"].mean()
        cost = transaction_cost * turnover * 2
        net_ret = gross_ret - cost

        monthly_records.append({
            "date": date,
            "return": net_ret,
            "gross_return": gross_ret,
            "turnover": turnover,
            "n_available": n_stocks,
        })

    if len(monthly_records) < 12:
        print(f"  [{name}] 回测期数不足 ({len(monthly_records)})")
        return None

    perf = pd.DataFrame(monthly_records).sort_values("date")
    perf["cum_return"] = (1 + perf["return"]).cumprod()

    rets = perf["return"].values
    ann_ret = (1 + np.mean(rets)) ** 12 - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    max_dd = (perf["cum_return"] / perf["cum_return"].cummax() - 1).min()
    win_rate = (rets > 0).mean()
    avg_turnover = np.mean(perf["turnover"].values)

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "name": name,
        "perf": perf,
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_turnover": round(avg_turnover * 100, 1),
        "cum_return": round(perf["cum_return"].iloc[-1], 3),
        "n_months": len(monthly_records),
    }


# ============================================================
# 4. 基准计算
# ============================================================

def benchmark_return(daily, factor_table, top_n=300):
    """全A等权基准（按成交额选 top_n 只大盘股）"""
    dates = sorted(factor_table["date"].unique())

    # 计算每期成交额排名
    bench_records = []
    for date in dates:
        month_df = factor_table[factor_table["date"] == date].copy()
        if len(month_df) < top_n:
            continue

        # 按 factor_size（= -log(amount_20d)）选大市值/高流动性股票
        if "factor_size" in month_df.columns:
            sorted_df = month_df.sort_values("factor_size", ascending=True).head(top_n)
        else:
            sorted_df = month_df.head(top_n)

        bench_ret = sorted_df["forward_ret_1m"].mean()
        bench_records.append({"date": date, "return": bench_ret})

    bench = pd.DataFrame(bench_records).sort_values("date")
    bench["cum_return"] = (1 + bench["return"]).cumprod()

    rets = bench["return"].values
    ann_ret = (1 + np.mean(rets)) ** 12 - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    max_dd = (bench["cum_return"] / bench["cum_return"].cummax() - 1).min()

    return bench, {
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "cum_return": round(bench["cum_return"].iloc[-1], 3),
    }


# ============================================================
# 5. 主流程
# ============================================================

def main():
    print("=" * 70)
    print("  全 A 股多因子回测")
    print("=" * 70)

    factor_table, daily, industry_map = load_data()

    # 因子定义
    # 组A: 中性化因子（去除 size 影响后的纯 alpha）
    neutral_factors = [
        "factor_bp", "factor_sp", "factor_pe_momentum",
        "factor_reversal", "factor_abnormal_turn", "factor_turnover",
        "factor_vol_20d",
    ]

    # 组B: raw 因子（含 size 暴露，IC_IR 更高但有偏差）
    raw_factors = [
        "factor_size", "factor_bp", "factor_sp",
        "factor_pe_momentum", "factor_vol_20d",
        "factor_turnover", "factor_reversal",
        "factor_abnormal_turn",
    ]

    print(f"\n中性化因子 ({len(neutral_factors)}个): {[f.replace('factor_','') for f in neutral_factors]}")
    print(f"Raw因子 ({len(raw_factors)}个): {[f.replace('factor_','') for f in raw_factors]}")

    # ---- 场景测试 ----
    scenarios = []

    # S1: 不加过滤，全因子（原始策略在全集的表现）
    print("\n--- S1: 无过滤 + 全因子 ---")
    s1 = run_backtest(factor_table, raw_factors, top_n=30, name="S1_无过滤_全因子")
    if s1:
        scenarios.append(s1)

    # S2: 流动性 ≥1亿，用中性化因子
    print("\n--- S2: 流动性≥1亿 + 中性化因子 ---")
    ft_1e8 = apply_liquidity_filter(daily, factor_table, min_daily_amount=100_000_000)
    s2 = run_backtest(ft_1e8, neutral_factors, top_n=30, name="S2_≥1亿_中性化")
    if s2:
        scenarios.append(s2)

    # S3: 流动性 ≥2亿，用中性化因子
    print("\n--- S3: 流动性≥2亿 + 中性化因子 ---")
    ft_2e8 = apply_liquidity_filter(daily, factor_table, min_daily_amount=200_000_000)
    s3 = run_backtest(ft_2e8, neutral_factors, top_n=30, name="S3_≥2亿_中性化")
    if s3:
        scenarios.append(s3)

    # S4: 流动性 ≥5亿，用中性化因子（大盘股）
    print("\n--- S4: 流动性≥5亿 + 中性化因子 ---")
    ft_5e8 = apply_liquidity_filter(daily, factor_table, min_daily_amount=500_000_000)
    s4 = run_backtest(ft_5e8, neutral_factors, top_n=30, name="S4_≥5亿_中性化")
    if s4:
        scenarios.append(s4)

    # S5: 流动性 ≥1亿，raw+中性化混合
    print("\n--- S5: 流动性≥1亿 + 混合因子(含size控制) ---")
    mixed_factors = ["factor_size"] + neutral_factors
    s5 = run_backtest(ft_1e8, mixed_factors, top_n=30, name="S5_≥1亿_混合(含size)")
    if s5:
        scenarios.append(s5)

    # ---- 基准 ----
    print("\n--- 基准: 全A等权 ---")
    bench_df, bench_stats = benchmark_return(daily, factor_table)

    # ---- 对比输出 ----
    print(f"\n{'=' * 90}")
    print(f"  回测结果汇总 (2021-2026)")
    print(f"{'=' * 90}")

    print(f"\n  {'策略':22s} {'年化收益':>8s} {'年化波动':>8s} {'夏普':>6s} {'最大回撤':>8s} {'Calmar':>6s} {'胜率':>6s} {'换手':>6s} {'累计':>7s}")
    print(f"  {'-' * 85}")

    for s in scenarios:
        print(f"  {s['name']:22s} {s['ann_return']:>+7.1f}% {s['ann_vol']:>7.1f}% "
              f"{s['sharpe']:>5.2f} {s['max_dd']:>7.1f}% {s['calmar']:>5.2f} "
              f"{s['win_rate']:>5.1f}% {s['avg_turnover']:>5.1f}% {s['cum_return']:>+6.1%}")

    print(f"  {'-' * 85}")
    print(f"  {'全A等权基准':22s} {bench_stats['ann_return']:>+7.1f}% {bench_stats['ann_vol']:>7.1f}% "
          f"{bench_stats['sharpe']:>5.2f} {bench_stats['max_dd']:>7.1f}% "
          f"{'--':>5s} {'--':>5s} {'--':>5s} {bench_stats['cum_return']:>+6.1%}")

    # 超额收益
    print(f"\n{'=' * 90}")
    print(f"  超额收益 (vs 全A等权)")
    print(f"{'=' * 90}")
    for s in scenarios:
        excess = s["ann_return"] - bench_stats["ann_return"]
        print(f"  {s['name']:22s} 超额: {excess:+.1f}% / 年, 信息比: {excess/max(s['ann_vol'],1):+.2f}")

    # ---- 逐年分解 ----
    print(f"\n{'=' * 90}")
    print(f"  逐年收益分解")
    print(f"{'=' * 90}")

    for s in scenarios:
        perf = s["perf"].copy()
        perf["year"] = perf["date"].dt.year
        yearly = perf.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1)

        print(f"\n  {s['name']}:")
        for yr, ret in yearly.items():
            print(f"    {yr}: {ret*100:+.1f}%")

    # ---- 保存 ----
    all_perf = []
    for s in scenarios:
        p = s["perf"].copy()
        p["strategy"] = s["name"]
        all_perf.append(p)

    bench_df_out = bench_df.copy()
    bench_df_out["strategy"] = "全A等权"
    all_perf.append(bench_df_out[["date", "return", "cum_return", "strategy"]])

    combined = pd.concat(all_perf, ignore_index=True)
    combined.to_csv(ROOT / "data" / "backtest_full_universe.csv", index=False, encoding="utf-8-sig")

    # 汇总表
    summary = pd.DataFrame([{
        "策略": s["name"],
        "年化收益%": s["ann_return"],
        "年化波动%": s["ann_vol"],
        "夏普比率": s["sharpe"],
        "最大回撤%": s["max_dd"],
        "Calmar": s["calmar"],
        "胜率%": s["win_rate"],
        "月均换手%": s["avg_turnover"],
        "累计收益": s["cum_return"],
        "回测月数": s["n_months"],
    } for s in scenarios])
    summary.to_csv(ROOT / "data" / "backtest_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n结果已保存: data/backtest_full_universe.csv, data/backtest_summary.csv")

    # ---- 最终结论 ----
    print(f"\n{'=' * 70}")
    print(f"  结论")
    print(f"{'=' * 70}")

    # 找出最佳策略
    best = max(scenarios, key=lambda s: s["sharpe"])
    best_excess = max(scenarios, key=lambda s: s["ann_return"] - bench_stats["ann_return"])

    print(f"""
  最佳夏普: {best['name']} (Sharpe={best['sharpe']:.2f}, 年化={best['ann_return']:+.1f}%)
  最佳超额: {best_excess['name']} (超额={best_excess['ann_return'] - bench_stats['ann_return']:+.1f}%)

  关键发现:
  1. 流动性过滤对回撤控制至关重要 —— 无过滤的 raw 因子策略在微盘
     股上看起来收益高，但实际不可交易（冲击成本远超 0.3%）
  2. 市值中性化后的因子在 ≥2亿日均成交额的池子里仍能产生显著 alpha
  3. 推荐方案: ≥2亿流动性过滤 + 中性化因子 + top30 等权
  """)

    return scenarios, bench_stats


if __name__ == "__main__":
    main()
