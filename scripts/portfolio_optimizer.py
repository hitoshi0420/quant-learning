"""
组合权重优化: 均值方差 / 风险平价 / 最小方差

在 top-N 选股基础上，用优化器分配个股权重，替代等权。
对比三种优化方法 + 等权基准。

用法:
    python portfolio_optimizer.py
    python portfolio_optimizer.py --plot
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import build_clean_factor_table, load_all
from data_fetcher import load_industries
from factor_test import ic_analysis
from multi_factor import select_factors, compute_composite_score


# ============================================================
# 0. 协方差估计 (Ledoit-Wolf 收缩)
# ============================================================

def estimate_covariance(returns: pd.DataFrame, shrinkage: float = 0.2) -> np.ndarray:
    """
    估计协方差矩阵，带收缩

    参数:
        returns: DataFrame [date, stock1, stock2, ...]，每格为该股票该期收益
        shrinkage: 收缩强度，0=样本协方差，1=对角矩阵
    """
    sample_cov = returns.cov().values
    # 收缩到对角矩阵 (constant correlation target)
    diag = np.diag(np.diag(sample_cov))
    avg_corr = 0.3  # A 股平均截面相关性
    sqrt_diag = np.sqrt(np.diag(sample_cov))
    target = diag + avg_corr * np.outer(sqrt_diag, sqrt_diag) * (1 - np.eye(len(diag)))
    shrunk = (1 - shrinkage) * sample_cov + shrinkage * target
    return shrunk


# ============================================================
# 1. 均值方差优化 (Max Sharpe)
# ============================================================

def max_sharpe_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_weight: float = 0.10,
    min_weight: float = 0.01,
) -> np.ndarray:
    """
    最大化 Sharpe ratio 的组合权重

    min  -mu^T w / sqrt(w^T Sigma w)
    s.t. sum(w) = 1, w_i in [min_weight, max_weight]

    参数:
        expected_returns: 预期收益向量 (n_stocks,)
        cov_matrix: 协方差矩阵 (n_stocks, n_stocks)
        max_weight: 单票最大权重
        min_weight: 单票最小权重

    返回: 权重向量
    """
    n = len(expected_returns)

    def neg_sharpe(w):
        port_ret = w @ expected_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        return -port_ret / port_vol if port_vol > 0 else 1e6

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n  # 等权起步

    result = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        return w / w.sum()
    else:
        return np.ones(n) / n


# ============================================================
# 2. 风险平价 (Equal Risk Contribution)
# ============================================================

def risk_parity_weights(
    cov_matrix: np.ndarray,
    max_weight: float = 0.10,
    min_weight: float = 0.01,
) -> np.ndarray:
    """
    等风险贡献权重

    min  sum_i sum_j (w_i * (Sigma w)_i - w_j * (Sigma w)_j)^2
    s.t. sum(w) = 1, w_i >= min_weight

    返回: 权重向量
    """
    n = cov_matrix.shape[0]

    def risk_concentration(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol == 0:
            return 1e6
        marginal_risk = cov_matrix @ w
        risk_contrib = w * marginal_risk / port_vol
        target = port_vol / n
        return np.sum((risk_contrib - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        risk_concentration, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        return w / w.sum()
    else:
        return np.ones(n) / n


# ============================================================
# 3. 最小方差 (Minimum Variance)
# ============================================================

def min_variance_weights(
    cov_matrix: np.ndarray,
    max_weight: float = 0.10,
    min_weight: float = 0.01,
) -> np.ndarray:
    """
    全局最小方差组合

    min  w^T Sigma w
    s.t. sum(w) = 1, w_i in [min_weight, max_weight]
    """
    n = cov_matrix.shape[0]

    def port_variance(w):
        return w @ cov_matrix @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        port_variance, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        return w / w.sum()
    else:
        return np.ones(n) / n


# ============================================================
# 4. 优化组合回测
# ============================================================

def backtest_optimized(
    factor_table: pd.DataFrame,
    score_col: str = "composite_score",
    top_n: int = 30,
    method: str = "equal",
    lookback: int = 12,
    max_weight: float = 0.10,
    min_weight: float = 0.01,
    transaction_cost: float = 0.003,
) -> dict:
    """
    用优化权重做月度调仓回测

    每月:
      1. 选 top_n 只得分最高的股票
      2. 用过去 lookback 个月的收益估计协方差 + 预期收益
      3. 优化权重
      4. 持有至下月

    参数:
        method: "equal" | "max_sharpe" | "risk_parity" | "min_variance"
        lookback: 协方差估计窗口 (月)
    """
    df = factor_table[["date", "code", score_col, "forward_ret_1m"]].dropna().copy()
    dates = sorted(df["date"].unique())

    # 构建历史收益矩阵（用于协方差估计）
    # 每期每只股票的实际收益
    ret_pivot = df.pivot_table(
        index="date", columns="code", values="forward_ret_1m"
    ).sort_index()

    prev_portfolio = {}
    monthly_rets = []

    for i, date in enumerate(tqdm(dates, desc=f"回测 {method}")):
        cross = df[df["date"] == date].copy()
        n_stocks = len(cross)
        if n_stocks < 10:
            continue

        # 选 top-N
        cross = cross.sort_values(score_col, ascending=False)
        selected = cross.head(min(top_n, n_stocks))
        selected_codes = selected["code"].tolist()

        # 获取历史收益
        if i >= lookback:
            hist = ret_pivot.iloc[i - lookback : i][selected_codes].dropna(axis=1)
        else:
            hist = ret_pivot.iloc[: i + 1][selected_codes].dropna(axis=1)

        common_cols = [c for c in selected_codes if c in hist.columns]
        if len(common_cols) < 5:
            weights = np.ones(len(selected_codes)) / len(selected_codes)
            actual_codes = selected_codes
        else:
            hist = hist[common_cols]
            cov = estimate_covariance(hist)

            if method == "equal":
                weights = np.ones(len(common_cols)) / len(common_cols)
            elif method == "max_sharpe":
                mu = hist.mean().values
                weights = max_sharpe_weights(mu, cov, max_weight, min_weight)
            elif method == "risk_parity":
                weights = risk_parity_weights(cov, max_weight, min_weight)
            elif method == "min_variance":
                weights = min_variance_weights(cov, max_weight, min_weight)
            else:
                weights = np.ones(len(common_cols)) / len(common_cols)

            actual_codes = common_cols

        # 构建权重字典
        current_portfolio = dict(zip(actual_codes, weights))

        # 组合收益
        selected_df = selected.set_index("code")
        port_ret = 0.0
        total_w = 0.0
        for code, w in current_portfolio.items():
            if code in selected_df.index:
                port_ret += w * selected_df.loc[code, "forward_ret_1m"]
                total_w += w

        if total_w > 0:
            port_ret /= total_w

        # 换手率
        if prev_portfolio:
            turnover = 0.0
            all_codes = set(list(prev_portfolio.keys()) + list(current_portfolio.keys()))
            for code in all_codes:
                w_old = prev_portfolio.get(code, 0)
                w_new = current_portfolio.get(code, 0)
                turnover += abs(w_new - w_old)
            turnover /= 2  # 双边换手
        else:
            turnover = 1.0

        prev_portfolio = current_portfolio

        net_ret = port_ret - transaction_cost * turnover * 2
        monthly_rets.append({
            "date": date,
            "return": net_ret,
            "gross_return": port_ret,
            "turnover": turnover,
            "n_selected": len(current_portfolio),
        })

    if not monthly_rets:
        raise ValueError(f"{method}: 回测期数不足")

    perf = pd.DataFrame(monthly_rets).sort_values("date")
    perf["cum_return"] = (1 + perf["return"]).cumprod()

    rets = perf["return"].values
    mean_ret = np.mean(rets)
    ann_ret = (1 + mean_ret) ** 12 - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    max_dd = (perf["cum_return"] / perf["cum_return"].cummax() - 1).min()
    win_rate = (rets > 0).mean()

    # 集中度
    weights_list = []
    for _, row in perf.iterrows():
        pass  # weights stored in monthly_rets but not in perf
    avg_turnover = perf["turnover"].mean()

    # 跟踪误差 (vs 等权)
    # will be computed in main

    print(f"\n  {method:16s}: 年化 {ann_ret*100:+.1f}%, "
          f"夏普 {sharpe:.2f}, 回撤 {max_dd*100:.1f}%, "
          f"胜率 {win_rate:.0%}, 换手 {avg_turnover:.0%}")

    return {
        "method": method,
        "perf": perf,
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 3),
        "avg_turnover": round(avg_turnover, 3),
        "cum_return": perf["cum_return"].iloc[-1],
    }


# ============================================================
# 5. 主流程
# ============================================================

def run_portfolio_optimization(plot: bool = False, top_n: int = 30):
    """对比四种权重方法"""
    print("=" * 60)
    print("组合权重优化: 等权 vs 均值方差 vs 风险平价 vs 最小方差")
    print("=" * 60)

    # 加载数据 + 因子合成
    print("\n[1/3] 数据准备...")
    factor_table = build_clean_factor_table(ROOT, start_date="2021-01-01",
                                             neutralize=False)
    ic_report = ic_analysis(factor_table)
    factor_cols = select_factors(factor_table, ic_report,
                                 min_ic_ir=0.10, min_t=1.5, prefer_raw=True)
    scored = compute_composite_score(factor_table, factor_cols, weighting="equal")

    print(f"\n[2/3] 对比四种权重方法 (top_{top_n})...")

    methods = [
        ("equal",        0.10, 0.00),
        ("max_sharpe",   0.10, 0.01),
        ("risk_parity",  0.10, 0.01),
        ("min_variance", 0.10, 0.01),
    ]

    results = []
    for method, max_w, min_w in methods:
        r = backtest_optimized(
            scored, top_n=top_n, method=method,
            max_weight=max_w, min_weight=min_w,
        )
        results.append(r)

    # 等权结果作为基准
    baseline = results[0]

    print(f"\n[3/3] 综合对比")
    print(f"\n{'=' * 70}")
    print(f"{'方法':16s} {'年化收益':>10s} {'夏普':>6s} {'回撤':>8s} {'胜率':>6s} {'换手':>6s} {'累计收益':>10s}")
    print(f"{'=' * 70}")

    for r in results:
        print(f"{r['method']:16s} {r['ann_return']:>+9.1f}% "
              f"{r['sharpe']:>6.2f} {r['max_dd']:>7.1f}% "
              f"{r['win_rate']:>6.0%} {r['avg_turnover']:>6.0%} "
              f"{(r['cum_return']-1)*100:>+9.1f}%")

    # 超额收益 (vs 等权)
    print(f"\n{'=' * 70}")
    print(f"vs 等权超额:")
    for r in results[1:]:
        excess_ret = r["ann_return"] - baseline["ann_return"]
        excess_sharpe = r["sharpe"] - baseline["sharpe"]
        print(f"  {r['method']:16s}: 超额收益 {excess_ret:+.1f}%, "
              f"夏普变化 {excess_sharpe:+.2f}")

    # 保存
    summary = pd.DataFrame([
        {"method": r["method"], "ann_return": r["ann_return"],
         "sharpe": r["sharpe"], "max_dd": r["max_dd"],
         "win_rate": r["win_rate"], "avg_turnover": r["avg_turnover"],
         "cum_return": r["cum_return"]}
        for r in results
    ])
    summary.to_csv(ROOT / "data" / "portfolio_optimization.csv",
                   index=False, encoding="utf-8-sig")

    # 保存各方法月度收益
    for r in results:
        r["perf"].to_csv(ROOT / "data" / f"portfolio_{r['method']}_monthly.csv",
                         index=False, encoding="utf-8-sig")

    print(f"\n报告已保存: data/portfolio_optimization.csv")

    if plot:
        plot_portfolio_comparison(results, ROOT / "data" / "plots")

    return results


def plot_portfolio_comparison(results: list, save_dir: Path):
    """四种权重方法对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    colors = {
        "equal": "#607d8b",
        "max_sharpe": "#e53935",
        "risk_parity": "#2196f3",
        "min_variance": "#4caf50",
    }
    labels_cn = {
        "equal": "等权基准",
        "max_sharpe": "最大夏普",
        "risk_parity": "风险平价",
        "min_variance": "最小方差",
    }

    # 图1: 净值曲线
    ax1 = fig.add_subplot(2, 3, 1)
    for r in results:
        perf = r["perf"]
        ax1.plot(perf["date"], perf["cum_return"],
                 color=colors.get(r["method"], "#999"),
                 linewidth=1.5,
                 label=f"{labels_cn[r['method']]} (夏普={r['sharpe']:.2f})")
    ax1.axhline(y=1, color="#999", linewidth=0.5, linestyle="--")
    ax1.set_title("净值曲线对比", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # 图2: 年化收益 vs 波动
    ax2 = fig.add_subplot(2, 3, 2)
    for r in results:
        ax2.scatter(r["ann_vol"], r["ann_return"],
                    color=colors.get(r["method"], "#999"),
                    s=120, zorder=5, label=labels_cn[r["method"]])
    ax2.set_xlabel("年化波动 (%)")
    ax2.set_ylabel("年化收益 (%)")
    ax2.set_title("风险-收益散点", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    # 等权基准连线
    if results:
        ax2.axhline(y=results[0]["ann_return"], color="#999",
                    linewidth=0.5, linestyle="--", alpha=0.5)

    # 图3: 回撤曲线
    ax3 = fig.add_subplot(2, 3, 3)
    for r in results:
        perf = r["perf"]
        dd = perf["cum_return"] / perf["cum_return"].cummax() - 1
        ax3.plot(perf["date"], dd * 100,
                 color=colors.get(r["method"], "#999"),
                 linewidth=1, alpha=0.7,
                 label=f"{labels_cn[r['method']]} (max={r['max_dd']:.1f}%)")
    ax3.set_title("回撤曲线对比", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # 图4: 月度换手率
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(len(results))
    methods_list = [r["method"] for r in results]
    turnovers = [r["avg_turnover"] * 100 for r in results]
    bars = ax4.bar(x, turnovers,
                   color=[colors.get(m, "#999") for m in methods_list])
    ax4.set_xticks(x)
    ax4.set_xticklabels([labels_cn[m] for m in methods_list], fontsize=9)
    ax4.set_title("月均换手率 (%)", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.2, axis="y")
    for bar, v in zip(bars, turnovers):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.0f}%", ha="center", fontsize=9)

    # 图5: 36 个月滚动夏普
    ax5 = fig.add_subplot(2, 3, 5)
    for r in results:
        perf = r["perf"]
        if len(perf) >= 36:
            roll_sharpe = perf["return"].rolling(36).apply(
                lambda x: (x.mean() * 12 - 0.02) / (x.std() * np.sqrt(12))
                if x.std() > 0 else np.nan
            )
            ax5.plot(perf["date"], roll_sharpe,
                     color=colors.get(r["method"], "#999"),
                     linewidth=1.5,
                     label=labels_cn[r["method"]])
    ax5.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax5.set_title("36月滚动夏普", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)

    # 图6: 绩效指标汇总表
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    table_data = []
    for r in results:
        table_data.append([
            labels_cn[r["method"]],
            f"{r['ann_return']:+.1f}%",
            f"{r['sharpe']:.2f}",
            f"{r['max_dd']:.1f}%",
            f"{r['win_rate']:.0%}",
        ])

    table = ax6.table(
        cellText=table_data,
        colLabels=["方法", "年化收益", "夏普", "最大回撤", "胜率"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for j in range(5):
        table[0, j].set_facecolor("#e53935")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax6.set_title("绩效指标汇总", fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_dir / "portfolio_comparison.png", dpi=150, bbox_inches="tight")
    print(f"图表已保存: {save_dir / 'portfolio_comparison.png'}")
    plt.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="组合权重优化")
    parser.add_argument("--plot", action="store_true", help="输出对比图")
    parser.add_argument("--top", type=int, default=30, help="持仓股票数")
    args = parser.parse_args()

    run_portfolio_optimization(plot=args.plot, top_n=args.top)
