"""
单因子检验框架

功能:
  - IC 分析（Rank IC 均值/IR/胜率/累计曲线）
  - 分层回测（5分位组合，多空收益）
  - 因子评估报告

用法:
    python factor_test.py              # 全因子评估报告
    python factor_test.py --plot       # 含图表输出
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


# ============================================================
# 1. IC 分析
# ============================================================

def ic_analysis(factor_table: pd.DataFrame) -> pd.DataFrame:
    """
    对每个因子做截面 Rank IC 分析

    IC_t = corr(factor_value_t, forward_return_t+1)  每期截面

    返回: 每个因子的 IC 统计
        ic_mean, ic_std, ic_ir, ic_pos_ratio, t_stat, p_value
    """
    factor_cols = [c for c in factor_table.columns if c.startswith("factor_")]
    results = []

    for col in tqdm(factor_cols, desc="IC 分析"):
        ic_series = []
        for date, grp in factor_table.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) < 10:
                continue
            ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
            ic_series.append({"date": date, "ic": ic})

        if not ic_series:
            continue

        ic_df = pd.DataFrame(ic_series)
        ic_vals = ic_df["ic"].dropna().values
        if len(ic_vals) < 5:
            continue

        ic_mean = np.mean(ic_vals)
        ic_std = np.std(ic_vals, ddof=1)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos = (ic_vals > 0).mean()
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_vals))) if ic_std > 0 else 0
        p_value = 2 * stats.t.sf(abs(t_stat), df=len(ic_vals) - 1)

        results.append({
            "factor": col.replace("factor_", ""),
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "ic_ir": round(ic_ir, 4),
            "ic_pos_ratio": round(ic_pos, 3),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "n_months": len(ic_vals),
            "ic_series": ic_vals,
        })

    report = pd.DataFrame(results).sort_values("ic_ir", ascending=False, key=abs)
    return report


# ============================================================
# 2. 分层回测
# ============================================================

def quantile_backtest(
    factor_table: pd.DataFrame,
    factor_name: str,
    n_quantiles: int = 5,
) -> dict:
    """
    对单个因子做分层回测

    每期按因子值分 n_quantiles 组，等权持有，计算各组的：
      - 累计收益
      - 年化收益
      - 年化波动
      - 夏普比率
      - 最大回撤

    返回: dict with group returns, cumulative, stats
    """
    col = f"factor_{factor_name}" if not factor_name.startswith("factor_") else factor_name
    if col not in factor_table.columns:
        raise ValueError(f"因子 {col} 不存在")

    df = factor_table[["date", "code", col, "forward_ret_1m"]].dropna().copy()

    # 每期按因子值分位
    df["quantile"] = df.groupby("date")[col].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["quantile"])

    # 各组等权收益
    group_returns = df.groupby(["date", "quantile"])["forward_ret_1m"].mean().reset_index()
    group_returns["quantile"] = group_returns["quantile"].astype(int)

    # 累计收益
    cumulative = {}
    group_stats = []
    for q in range(n_quantiles):
        g = group_returns[group_returns["quantile"] == q].sort_values("date")
        g["cum_ret"] = (1 + g["forward_ret_1m"]).cumprod()
        cumulative[q] = g

        rets = g["forward_ret_1m"].values
        ann_ret = (1 + np.mean(rets)) ** 12 - 1
        ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum_max = g["cum_ret"].cummax()
        drawdown = (g["cum_ret"] - cum_max) / cum_max
        max_dd = drawdown.min()

        group_stats.append({
            "quantile": q + 1,
            "label": ["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"][q],
            "ann_return": round(ann_ret * 100, 2),
            "ann_vol": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_dd": round(max_dd * 100, 2),
        })

    # 多空收益 (Q5 - Q1)
    if 0 in cumulative and n_quantiles - 1 in cumulative:
        long_ret = cumulative[n_quantiles - 1]["forward_ret_1m"].values
        short_ret = cumulative[0]["forward_ret_1m"].values
        spread_ret = long_ret - short_ret
        ann_spread = (1 + np.mean(spread_ret)) ** 12 - 1
        ann_spread_vol = np.std(spread_ret, ddof=1) * np.sqrt(12)
        spread_sharpe = ann_spread / ann_spread_vol if ann_spread_vol > 0 else 0
    else:
        ann_spread = ann_spread_vol = spread_sharpe = 0

    return {
        "factor": factor_name,
        "group_returns": group_returns,
        "cumulative": cumulative,
        "group_stats": group_stats,
        "spread_ann_return": round(ann_spread * 100, 2),
        "spread_sharpe": round(spread_sharpe, 2),
        "monotonic": _check_monotonic(group_stats),
    }


def _check_monotonic(group_stats: list) -> bool:
    rets = [g["ann_return"] for g in group_stats]
    # Q1 到 Q5 收益应单调递增（或递减）
    increasing = all(rets[i] <= rets[i + 1] for i in range(len(rets) - 1))
    decreasing = all(rets[i] >= rets[i + 1] for i in range(len(rets) - 1))
    return increasing or decreasing


# ============================================================
# 3. 综合评估报告
# ============================================================

def full_report(factor_table: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    全因子评估报告

    包含 IC 分析 + 分层回测，返回综合评分表
    """
    print("=" * 60)
    print("多因子评估报告")
    print("=" * 60)

    # IC 分析
    ic_report = ic_analysis(factor_table)

    # 分层回测
    backtest_results = []
    for _, row in tqdm(ic_report.iterrows(), total=len(ic_report), desc="分层回测"):
        try:
            bt = quantile_backtest(factor_table, row["factor"])
            backtest_results.append({
                "factor": row["factor"],
                "ic_mean": row["ic_mean"],
                "ic_ir": row["ic_ir"],
                "ic_pos_ratio": row["ic_pos_ratio"],
                "t_stat": row["t_stat"],
                "p_value": row["p_value"],
                "long_ret": bt["group_stats"][-1]["ann_return"],
                "long_sharpe": bt["group_stats"][-1]["sharpe"],
                "short_ret": bt["group_stats"][0]["ann_return"],
                "spread_ret": bt["spread_ann_return"],
                "spread_sharpe": bt["spread_sharpe"],
                "monotonic": bt["monotonic"],
            })
        except Exception as e:
            print(f"  {row['factor']} 分层回测失败: {e}")

    report = pd.DataFrame(backtest_results).sort_values("ic_ir", ascending=False, key=abs)

    # 评分（综合 IC_IR × 显著性 + 分层单调性）
    report["score"] = (
        abs(report["ic_ir"]) * 0.5
        + (report["p_value"] < 0.05).astype(float) * 2
        + report["monotonic"].astype(float) * 3
        + abs(report["spread_sharpe"]).clip(0, 2) * 0.5
    )
    report = report.sort_values("score", ascending=False)

    print("\n" + "=" * 60)
    print("因子排名 (综合评分)")
    print("=" * 60)

    cols_show = ["factor", "ic_mean", "ic_ir", "t_stat", "long_ret", "spread_ret",
                 "spread_sharpe", "monotonic", "score"]
    print(report[cols_show].head(top_n).to_string(index=False))

    # 有效因子判断
    valid_factors = report[
        (abs(report["ic_ir"]) > 0.1) | (report["p_value"] < 0.1)
    ]
    print(f"\n有效因子: {len(valid_factors)} / {len(report)} 个")
    print(f"  IC 显著 (|t|>2):  {(abs(report['t_stat'])>2).sum()} 个")
    print(f"  分层单调:         {report['monotonic'].sum()} 个")

    return report


# ============================================================
# 4. 可视化
# ============================================================

def plot_ic_curve(ic_report: pd.DataFrame, save_dir: Path | None = None):
    """绘制 IC 累计曲线"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    top_factors = ic_report.head(6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_factors.iterrows()):
        if idx >= 6:
            break
        ax = axes[idx]
        ic_vals = row["ic_series"]
        cum_ic = np.cumsum(ic_vals)
        ax.plot(range(len(cum_ic)), cum_ic, color="#e53935", linewidth=1.5)
        ax.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
        ax.set_title(
            f"{row['factor']} (IR={row['ic_ir']:.2f}, t={row['t_stat']:.1f})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "ic_cumulative.png", dpi=150, bbox_inches="tight")
        print(f"IC 图保存至: {save_dir / 'ic_cumulative.png'}")
    plt.close()


def plot_quantile_returns(bt_results: list, save_dir: Path | None = None):
    """绘制分层回测累计收益"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = ["#2196f3", "#64b5f6", "#9e9e9e", "#ef9a9a", "#e53935"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, bt in enumerate(bt_results[:6]):
        ax = axes[idx]
        for q in range(5):
            if q in bt["cumulative"]:
                g = bt["cumulative"][q]
                ax.plot(
                    range(len(g)), g["cum_ret"].values,
                    color=colors[q], linewidth=1.2,
                    label=f"Q{q + 1}",
                )
        ax.set_title(f"{bt['factor']} (多空={bt['spread_sharpe']:.2f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "quantile_returns.png", dpi=150, bbox_inches="tight")
        print(f"分层回测图保存至: {save_dir / 'quantile_returns.png'}")
    plt.close()


# ============================================================
# 5. CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="单因子检验")
    parser.add_argument("--plot", action="store_true", help="输出图表")
    parser.add_argument("--factor", type=str, default="", help="指定因子名")
    args = parser.parse_args()

    # 构建因子表
    print("构建因子表...")
    df = build_clean_factor_table(ROOT, start_date="2021-01-01")

    # 评估报告
    report = full_report(df)

    # 保存报告
    report_path = ROOT / "data" / "factor_report.csv"
    # 去掉 ic_series 列
    save_cols = [c for c in report.columns if c != "ic_series"]
    report[save_cols].to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: {report_path}")

    # 图表
    if args.plot:
        img_dir = ROOT / "data" / "plots"
        plot_ic_curve(report, img_dir)
        top_factors = report.head(6)["factor"].tolist()
        bt_list = [quantile_backtest(df, f) for f in top_factors]
        plot_quantile_returns(bt_list, img_dir)

    # 单个因子详情
    if args.factor:
        print(f"\n{'=' * 60}")
        print(f"因子 {args.factor} 详情")
        print(f"{'=' * 60}")
        bt = quantile_backtest(df, args.factor)
        print("\n分层统计:")
        for g in bt["group_stats"]:
            print(f"  {g['label']}: 年化 {g['ann_return']:+.1f}%, 夏普 {g['sharpe']:.2f}, 回撤 {g['max_dd']:.1f}%")
        print(f"\n多空收益: {bt['spread_ann_return']:+.1f}%, 夏普 {bt['spread_sharpe']:.2f}")
        print(f"分层单调: {'是' if bt['monotonic'] else '否'}")
