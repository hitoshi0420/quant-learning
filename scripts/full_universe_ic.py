"""
全 A 股多因子 IC 分析（带流动性过滤）
用法: python scripts/full_universe_ic.py
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import build_clean_factor_table

# ============================================================
# 1. 流动性过滤：计算每只股票的日均成交额
# ============================================================

def compute_liquidity_filter(min_daily_amount: float = 30_000_000,  # 3000万
                             min_days: int = 60) -> set:
    """
    扫描所有日线文件，返回满足流动性要求的股票代码集合

    条件: 近一年日均成交额 >= min_daily_amount，且至少有 min_days 个交易日
    """
    daily_dir = ROOT / "data" / "daily"
    qualified = {}
    excluded = {}

    files = sorted(daily_dir.glob("*.parquet"))
    print(f"扫描 {len(files)} 只股票的流动性...")

    for f in tqdm(files):
        try:
            df = pd.read_parquet(f, columns=["date", "amount"])
            df["date"] = pd.to_datetime(df["date"])
            # 只看最近一年
            recent = df[df["date"] >= df["date"].max() - pd.DateOffset(years=1)]
            if len(recent) < min_days:
                excluded[f.stem] = "交易日不足"
                continue
            avg_amount = recent["amount"].mean()
            if avg_amount >= min_daily_amount:
                qualified[f.stem] = round(avg_amount / 1e8, 2)
            else:
                excluded[f.stem] = f"日均{avg_amount/1e4:.0f}万"
        except Exception as e:
            excluded[f.stem] = str(e)

    print(f"\n流动性过滤结果:")
    print(f"  合格 (日均≥{min_daily_amount/1e8:.1f}亿): {len(qualified)} 只")
    print(f"  排除: {len(excluded)} 只")

    # 按成交额分档
    if qualified:
        amounts = list(qualified.values())
        print(f"  成交额分布: 最小={min(amounts):.2f}亿, 中位数={np.median(amounts):.2f}亿, 最大={max(amounts):.2f}亿")

    return set(qualified.keys()), qualified, excluded


# ============================================================
# 2. IC 分析
# ============================================================

def ic_analysis_full(factor_table: pd.DataFrame) -> pd.DataFrame:
    """对每个因子做截面 Rank IC 分析"""
    factor_cols = [c for c in factor_table.columns if c.startswith("factor_")]
    results = []

    for col in tqdm(factor_cols, desc="IC 分析"):
        ic_series = []
        for date, grp in factor_table.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) < 10:
                continue
            ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
            if not np.isnan(ic):
                ic_series.append({"date": date, "ic": ic})

        if len(ic_series) < 6:
            continue

        ic_df = pd.DataFrame(ic_series)
        ic_vals = ic_df["ic"].dropna().values

        ic_mean = np.mean(ic_vals)
        ic_std = np.std(ic_vals, ddof=1)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos = (ic_vals > 0).mean()
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_vals))) if ic_std > 0 else 0
        p_value = 2 * stats.t.sf(abs(t_stat), df=len(ic_vals) - 1)

        # IC 衰减：最近 12 个月的 IC 均值
        ic_recent = np.mean(ic_vals[-12:]) if len(ic_vals) >= 12 else ic_mean

        results.append({
            "factor": col.replace("factor_", ""),
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "ic_ir": round(ic_ir, 4),
            "ic_pos_ratio": round(ic_pos, 3),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "n_months": len(ic_vals),
            "ic_recent_12m": round(ic_recent, 4),
            "significant": abs(t_stat) > 2.0 or p_value < 0.05,
        })

    return pd.DataFrame(results).sort_values("ic_ir", ascending=False, key=abs)


# ============================================================
# 3. 分层回测
# ============================================================

def quantile_backtest_quick(factor_table: pd.DataFrame, factor_name: str, n_quantiles: int = 5):
    """快速分层回测"""
    col = f"factor_{factor_name}" if not factor_name.startswith("factor_") else factor_name
    if col not in factor_table.columns:
        return None

    df = factor_table[["date", "code", col, "forward_ret_1m"]].dropna().copy()

    try:
        df["quantile"] = df.groupby("date")[col].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
        )
    except Exception:
        return None

    df = df.dropna(subset=["quantile"])
    group_returns = df.groupby(["date", "quantile"])["forward_ret_1m"].mean().reset_index()

    stats_by_q = []
    for q in range(n_quantiles):
        g = group_returns[group_returns["quantile"] == q]
        if len(g) == 0:
            continue
        rets = g["forward_ret_1m"].values
        ann_ret = (1 + np.mean(rets)) ** 12 - 1
        ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + g["forward_ret_1m"]).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        stats_by_q.append({
            "quantile": q + 1,
            "ann_return": round(ann_ret * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_dd": round(max_dd * 100, 2),
        })

    if len(stats_by_q) < 2:
        return None

    # 多空
    top_ret = stats_by_q[-1]["ann_return"]
    bottom_ret = stats_by_q[0]["ann_return"]
    spread_ret = top_ret - bottom_ret

    # 单调性
    rets_q = [s["ann_return"] for s in stats_by_q]
    monotonic = all(rets_q[i] <= rets_q[i+1] for i in range(len(rets_q)-1)) or \
                all(rets_q[i] >= rets_q[i+1] for i in range(len(rets_q)-1))

    return {
        "spread_ret": round(spread_ret, 2),
        "top_ret": top_ret,
        "bottom_ret": bottom_ret,
        "monotonic": monotonic,
        "stats_by_q": stats_by_q,
    }


# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 70)
    print("  全 A 股多因子 IC 分析（含流动性过滤）")
    print("=" * 70)

    # ---- 流动性过滤 ----
    qualified_codes, amounts, excluded = compute_liquidity_filter(
        min_daily_amount=30_000_000
    )

    if len(qualified_codes) < 100:
        print("合格股票太少，降低门槛重试...")
        qualified_codes, amounts, excluded = compute_liquidity_filter(
            min_daily_amount=10_000_000
        )

    # 保存过滤结果
    pd.DataFrame([
        {"code": c, "avg_amount_yi": a} for c, a in sorted(amounts.items(), key=lambda x: -x[1])
    ]).to_csv(ROOT / "data" / "liquidity_filter.csv", index=False, encoding="utf-8-sig")

    # ---- 构建因子表 ----
    print(f"\n[2/4] 构建月度因子表（{len(qualified_codes)} 只股票）...")

    # 构建完整因子表（全A股）—— 缓存到 parquet 避免重复计算
    cache_path = ROOT / "data" / "factor_table_full.parquet"
    if cache_path.exists():
        print("  加载缓存因子表...")
        full_table = pd.read_parquet(cache_path)
        full_table["date"] = pd.to_datetime(full_table["date"])
    else:
        full_table = build_clean_factor_table(ROOT, start_date="2021-01-01")
        full_table.to_parquet(cache_path, index=False)
        print(f"  因子表已缓存: {cache_path}")

    # 过滤到合格股票
    factor_table = full_table[full_table["code"].isin(qualified_codes)].copy()

    n_months = factor_table["date"].nunique()
    n_stocks = factor_table["code"].nunique()
    n_factors = len([c for c in factor_table.columns if c.startswith("factor_")])
    print(f"  过滤后: {n_months} 个月, {n_stocks} 只股票, {n_factors} 个因子")

    # ---- IC 分析 ----
    print(f"\n[3/4] IC 分析...")
    ic_report = ic_analysis_full(factor_table)

    # ---- 分层回测 ----
    print(f"\n[4/4] 分层回测 (Top 15 因子)...")
    top_factors = ic_report.head(15)

    bt_results = []
    for _, row in tqdm(top_factors.iterrows(), total=len(top_factors), desc="分层回测"):
        bt = quantile_backtest_quick(factor_table, row["factor"])
        if bt:
            bt_results.append({
                "factor": row["factor"],
                "ic_mean": row["ic_mean"],
                "ic_ir": row["ic_ir"],
                "t_stat": row["t_stat"],
                "p_value": row["p_value"],
                "ic_recent_12m": row["ic_recent_12m"],
                "spread_ret": bt["spread_ret"],
                "top_ret": bt["top_ret"],
                "bottom_ret": bt["bottom_ret"],
                "monotonic": bt["monotonic"],
            })

    bt_df = pd.DataFrame(bt_results)

    # 综合评分
    bt_df["score"] = (
        abs(bt_df["ic_ir"]) * 0.5
        + (bt_df["p_value"] < 0.05).astype(float) * 2
        + bt_df["monotonic"].astype(float) * 3
        + abs(bt_df["spread_ret"]).clip(0, 30) / 30 * 2
    )
    bt_df = bt_df.sort_values("score", ascending=False)

    # ---- 对比旧报告 ----
    old_report_path = ROOT / "data" / "factor_report.csv"
    has_old = old_report_path.exists()

    # ---- 输出 ----
    print(f"\n{'=' * 70}")
    print(f"  全 A 股因子排名 (流动性过滤后: {n_stocks} 只)")
    print(f"{'=' * 70}")

    print(f"\n  {'因子':28s} {'IC均值':>7s} {'IC_IR':>7s} {'T值':>7s} {'P值':>7s} {'分层多空':>8s} {'单调':>4s} {'评分':>6s}")
    print(f"  {'-' * 80}")
    for _, row in bt_df.iterrows():
        name = row["factor"].replace("_raw", "")
        mono = "✓" if row["monotonic"] else ""
        print(f"  {name:28s} {row['ic_mean']:>+7.4f} {row['ic_ir']:>+7.3f} "
              f"{row['t_stat']:>+7.2f} {row['p_value']:>7.4f} "
              f"{row['spread_ret']:>+7.1f}% {mono:>4s} {row['score']:>5.1f}")

    # 有效因子统计
    valid_factors = bt_df[bt_df["p_value"] < 0.10]
    monotonic_factors = bt_df[bt_df["monotonic"]]
    strong_factors = bt_df[abs(bt_df["t_stat"]) > 2.0]

    print(f"\n  显著因子 (p<0.10): {len(valid_factors)} 个")
    print(f"  分层单调: {len(monotonic_factors)} 个")
    print(f"  |t|>2: {len(strong_factors)} 个")

    # ---- 对比旧报告 ----
    if has_old:
        old = pd.read_csv(old_report_path)
        print(f"\n{'=' * 70}")
        print(f"  新旧对比 (旧=沪深300约200只, 新=全A过滤后{n_stocks}只)")
        print(f"{'=' * 70}")

        # 匹配因子
        old_factors = set(old["factor"].tolist())
        new_factors = set(bt_df["factor"].tolist())
        common = old_factors & new_factors

        print(f"\n  {'因子':28s} {'旧IC_IR':>8s} {'新IC_IR':>8s} {'变化':>8s} {'状态':>10s}")
        print(f"  {'-' * 70}")

        old_ir_map = dict(zip(old["factor"], old["ic_ir"]))
        new_ir_map = dict(zip(bt_df["factor"], bt_df["ic_ir"]))

        comparisons = []
        for f in common:
            old_ir = old_ir_map.get(f, 0)
            new_ir = new_ir_map.get(f, 0)
            delta = new_ir - old_ir
            comparisons.append((f, old_ir, new_ir, delta))

        # 还包含新的显著因子
        for f in new_factors - common:
            new_ir = new_ir_map.get(f, 0)
            comparisons.append((f, 0, new_ir, new_ir))

        comparisons.sort(key=lambda x: abs(x[3]), reverse=True)

        for f, old_ir, new_ir, delta in comparisons[:20]:
            name = f.replace("_raw", "")
            status = "增强 ↑" if delta > 0.02 else ("减弱 ↓" if delta < -0.02 else "持平")
            print(f"  {name:28s} {old_ir:>+8.3f} {new_ir:>+8.3f} {delta:>+8.3f} {status:>10s}")

        # 失效因子
        dropped = old_factors - new_factors
        if dropped:
            print(f"\n  失效因子 (旧有但新报告未入Top15): {len(dropped)} 个")
            for f in list(dropped)[:10]:
                old_ir = old_ir_map.get(f, 0)
                print(f"    {f.replace('_raw', ''):30s} 旧IC_IR={old_ir:+.3f}")

        # 新有效因子
        new_significant = new_factors - old_factors
        if new_significant:
            print(f"\n  新显著因子 (全A股新增): {len(new_significant)} 个")
            for f in list(new_significant)[:10]:
                new_ir = new_ir_map.get(f, 0)
                name = f.replace("_raw", "")
                print(f"    {name:30s} 新IC_IR={new_ir:+.3f}")

    # ---- 保存 ----
    bt_df.to_csv(ROOT / "data" / "factor_report_full.csv", index=False, encoding="utf-8-sig")
    ic_report.to_csv(ROOT / "data" / "ic_report_full.csv", index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: data/factor_report_full.csv, data/ic_report_full.csv")

    # ---- 结论 ----
    print(f"\n{'=' * 70}")
    print(f"  结论")
    print(f"{'=' * 70}")

    n_effective = len(bt_df[abs(bt_df["ic_ir"]) > 0.10])
    n_strong = len(bt_df[(abs(bt_df["ic_ir"]) > 0.15) & (bt_df["monotonic"])])

    print(f"""
  1. 流动性过滤: {len(qualified_codes)}/{len(qualified_codes)+len(excluded)} 只股票合格（日均成交≥3000万）
  2. 有效因子 (|IC_IR|>0.10): {n_effective}/{n_factors} 个
  3. 强因子 (|IC_IR|>0.15 且分层单调): {n_strong} 个
  4. 策略建议: {"因子框架仍然有效，建议使用流动性过滤后的选股池" if n_effective >= 5 else "因子有效性在大pool中显著下降，需要重新筛选因子或降低阈值"}
  """)

    return bt_df, ic_report

if __name__ == "__main__":
    main()
