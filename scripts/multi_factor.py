"""
多因子合成 + 选股回测

流程:
  1. 从单因子检验报告中筛选有效因子
  2. 等权 / IC_IR 加权合成综合得分
  3. 每月按得分选前 N 只股票，等权持有
  4. 回测对比沪深300基准

用法:
    python multi_factor.py              # 运行回测
    python multi_factor.py --plot       # 含净值曲线图
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import build_clean_factor_table, load_all, winsorize
from factor_test import ic_analysis


# ============================================================
# 1. 因子筛选
# ============================================================

def select_factors(factor_table: pd.DataFrame, ic_report: pd.DataFrame,
                   min_ic_ir: float = 0.10, min_t: float = 1.5,
                   prefer_raw: bool = False) -> list:
    """
    从 IC 报告中筛选有效因子

    策略：
      - |IC_IR| > min_ic_ir 或 |t| > min_t
      - prefer_raw: 优先选择 raw 版本（未中性化）
      - 排除与其他因子高度相关的冗余因子

    返回: 选中的因子列名列表
    """
    # 筛选显著因子
    significant = ic_report[
        (abs(ic_report["ic_ir"]) > min_ic_ir) |
        (abs(ic_report["t_stat"]) > min_t)
    ].copy()

    print(f"显著因子: {len(significant)} / {len(ic_report)} 个")

    # 去重：如果同时有 raw 和中性化版本，按 prefer_raw 选择
    factor_names = significant["factor"].tolist()

    # 剔除 volatility_20d 和 volatility_60d 中的 raw（与 vol_20d_raw / vol_60d_raw 重复）
    selected = []
    seen_base = set()

    for f in factor_names:
        base = f.replace("_raw", "")
        if base in seen_base:
            continue
        # 如果同时有 raw 和中性化版本，按偏好选择
        raw_name = f"{base}_raw"
        neutral_name = base
        has_raw = raw_name in factor_names
        has_neutral = neutral_name in factor_names

        if has_raw and has_neutral:
            chosen = raw_name if prefer_raw else neutral_name
        elif has_raw:
            chosen = raw_name
        else:
            chosen = f

        if chosen not in selected:
            selected.append(chosen)
            seen_base.add(base)

    # 只保留在 factor_table 中存在的列
    available = [c for c in factor_table.columns if c.startswith("factor_")]
    selected = [f for f in selected if f"factor_{f}" in available
                or (f.startswith("factor_") and f in available)]

    # 统一加 factor_ 前缀
    result = []
    for f in selected:
        full = f if f.startswith("factor_") else f"factor_{f}"
        if full in available:
            result.append(full)

    print(f"去重后: {len(result)} 个因子: {result}")
    return result


# ============================================================
# 2. 综合得分计算
# ============================================================

def compute_composite_score(factor_table: pd.DataFrame,
                            factor_cols: list,
                            weighting: str = "equal") -> pd.DataFrame:
    """
    计算综合得分

    weighting:
      - "equal": 等权
      - "ic_ir": IC_IR 加权（需先跑 IC 分析）
    """
    df = factor_table.copy()

    # 有效性检查
    available = [c for c in factor_cols if c in df.columns]
    if not available:
        raise ValueError("没有可用的因子列")

    # 统一方向：IC 为负的因子取反
    ic_vals = {}
    for col in available:
        for date, grp in df.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) >= 10:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(valid[col], valid["forward_ret_1m"])
                ic_vals[col] = ic
                break

    for col, ic in ic_vals.items():
        if ic < 0:
            df[col] = -df[col]

    # 合成
    if weighting == "ic_ir":
        # 计算 IC_IR 作为权重
        weights = {}
        for col in available:
            ic_list = []
            for date, grp in df.groupby("date"):
                valid = grp[[col, "forward_ret_1m"]].dropna()
                if len(valid) >= 10:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(valid[col], valid["forward_ret_1m"])
                    ic_list.append(abs(ic))
            if ic_list:
                ic_mean = np.mean(ic_list)
                ic_std = np.std(ic_list, ddof=1)
                weights[col] = abs(ic_mean / ic_std) if ic_std > 0 else 0
            else:
                weights[col] = 0

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        print("IC_IR 权重:")
        for col, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {col.replace('factor_', ''):25s}: {w:.3f}")

        df["composite_score"] = sum(
            df[c].fillna(0) * w for c, w in weights.items()
        )
    else:
        # 等权
        df["composite_score"] = sum(
            df[c].fillna(0) for c in available
        ) / len(available)

    return df


# ============================================================
# 3. 选股组合回测
# ============================================================

def backtest_portfolio(
    factor_table: pd.DataFrame,
    score_col: str = "composite_score",
    top_pct: float = 0.2,
    top_n: int = 0,
    transaction_cost: float = 0.003,  # 0.3% 单边
) -> dict:
    """
    月度调仓回测

    每月选得分最高的 top_pct 或 top_n 只股票，等权持有
    含交易成本（佣金 + 印花税 + 滑点）

    返回: 月度收益序列、累计净值、绩效指标
    """
    df = factor_table[["date", "code", score_col, "forward_ret_1m"]].dropna().copy()

    # 每月选股
    monthly_ret = []
    turnover_list = []
    prev_portfolio = set()

    for date in sorted(df["date"].unique()):
        month_data = df[df["date"] == date].copy()
        n_stocks = len(month_data)
        if n_stocks < 10:
            continue

        # 选前 N 只
        if top_n > 0:
            n_select = min(top_n, n_stocks)
        else:
            n_select = max(int(n_stocks * top_pct), 10)

        month_data = month_data.sort_values(score_col, ascending=False)
        selected = month_data.head(n_select)

        # 换手率
        current_portfolio = set(selected["code"].tolist())
        if prev_portfolio:
            turnover = len(current_portfolio - prev_portfolio) / len(current_portfolio)
        else:
            turnover = 1.0
        turnover_list.append(turnover)
        prev_portfolio = current_portfolio

        # 等权收益（扣除交易成本）
        gross_ret = selected["forward_ret_1m"].mean()
        # 换手产生的交易成本
        cost = transaction_cost * turnover * 2  # 买入+卖出
        net_ret = gross_ret - cost

        monthly_ret.append({
            "date": date,
            "return": net_ret,
            "gross_return": gross_ret,
            "turnover": turnover,
            "n_selected": n_select,
            "n_total": n_stocks,
        })

    if not monthly_ret:
        raise ValueError("回测期数不足")

    perf = pd.DataFrame(monthly_ret).sort_values("date")
    perf["cum_return"] = (1 + perf["return"]).cumprod()

    # --- 绩效指标 ---
    rets = perf["return"].values
    n_months = len(rets)
    ann_ret = (1 + np.mean(rets)) ** 12 - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0  # 2% 无风险利率
    max_dd = (perf["cum_return"] / perf["cum_return"].cummax() - 1).min()
    win_rate = (rets > 0).mean()
    avg_turnover = np.mean(turnover_list)

    # 盈亏比
    pos_rets = rets[rets > 0]
    neg_rets = rets[rets < 0]
    profit_loss = abs(pos_rets.mean() / neg_rets.mean()) if len(neg_rets) > 0 else np.inf

    print(f"\n{'=' * 50}")
    print(f"回测结果 ({n_months} 个月)")
    print(f"{'=' * 50}")
    print(f"年化收益:     {ann_ret * 100:+.1f}%")
    print(f"年化波动:     {ann_vol * 100:.1f}%")
    print(f"夏普比率:     {sharpe:.2f}")
    print(f"最大回撤:     {max_dd * 100:.1f}%")
    print(f"胜率:         {win_rate * 100:.0f}%")
    print(f"盈亏比:       {profit_loss:.2f}")
    print(f"月均换手:     {avg_turnover * 100:.0f}%")
    print(f"月均选股:     {perf['n_selected'].mean():.0f} / {perf['n_total'].mean():.0f} 只")

    return {
        "perf": perf,
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 3),
        "profit_loss": round(profit_loss, 2),
        "avg_turnover": round(avg_turnover, 3),
        "cum_return": perf["cum_return"].iloc[-1],
    }


# ============================================================
# 4. 基准对比
# ============================================================

def benchmark_return(daily: pd.DataFrame, start_date: str = "2021-01-01") -> pd.DataFrame:
    """沪深300 等权基准月度收益"""
    df = daily.copy()
    df = df[df["date"] >= pd.Timestamp(start_date)]

    # 月度收益：每月第一天到最后一天
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_close = df.groupby(["year_month", "code"])["close"].last().reset_index()
    monthly_close["next_close"] = monthly_close.groupby("code")["close"].shift(-1)
    monthly_close["ret"] = monthly_close["next_close"] / monthly_close["close"] - 1
    monthly_close = monthly_close.dropna(subset=["ret"])

    # 等权
    benchmark = monthly_close.groupby("year_month")["ret"].mean().reset_index()
    benchmark["date"] = benchmark["year_month"].dt.to_timestamp()
    benchmark = benchmark.sort_values("date")
    benchmark["cum_return"] = (1 + benchmark["ret"]).cumprod()

    ann_ret = (1 + benchmark["ret"].mean()) ** 12 - 1
    ann_vol = benchmark["ret"].std() * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    max_dd = (benchmark["cum_return"] / benchmark["cum_return"].cummax() - 1).min()

    print(f"\n沪深300 等权基准:")
    print(f"  年化收益: {ann_ret * 100:+.1f}%, 夏普: {sharpe:.2f}, 最大回撤: {max_dd * 100:.1f}%")

    return benchmark


# ============================================================
# 5. 主流程
# ============================================================

def run_backtest(
    start_date: str = "2021-01-01",
    top_pct: float = 0.2,
    min_ic_ir: float = 0.10,
    top_n: int = 0,
    weighting: str = "equal",
    plot: bool = False,
):
    """一键运行完整回测"""
    print("=" * 60)
    print("多因子选股回测")
    print("=" * 60)

    # 1. 加载数据
    daily, fin, industry = load_all(ROOT)
    factor_table = build_clean_factor_table(ROOT, start_date=start_date)

    # 2. 因子筛选
    print("\n[1/4] 因子筛选...")
    ic_report = ic_analysis(factor_table)
    factor_cols = select_factors(factor_table, ic_report,
                                 min_ic_ir=min_ic_ir, prefer_raw=True)

    if len(factor_cols) < 3:
        # 兜底：用 top 5 IC_IR 最高的因子
        top5 = ic_report.head(5)["factor"].tolist()
        factor_cols = [f"factor_{f}" if not f.startswith("factor_") else f
                       for f in top5]
        factor_cols = [f for f in factor_cols if f in factor_table.columns]
        print(f"因子太少，改用 Top 5: {factor_cols}")

    # 3. 合成得分
    print("\n[2/4] 因子合成...")
    scored = compute_composite_score(factor_table, factor_cols, weighting=weighting)

    # 4. 回测
    print("\n[3/4] 组合回测...")
    result = backtest_portfolio(scored, top_pct=top_pct, top_n=top_n)

    # 5. 基准对比
    print("\n[4/4] 基准对比...")
    benchmark = benchmark_return(daily, start_date)

    # 超额收益
    perf = result["perf"]
    common_dates = set(perf["date"].dt.to_period("M")) & set(
        benchmark["date"].dt.to_period("M")
    )
    excess = result["ann_return"] - (
        (1 + benchmark["ret"].mean()) ** 12 - 1
    ) * 100
    print(f"\n超额收益 (vs 等权HS300): {excess:+.1f}%")

    # 6. 图表
    if plot:
        plot_results(perf, benchmark, factor_cols, ROOT / "data" / "plots",
                     ann_ret=result["ann_return"], sharpe=result["sharpe"])

    return result, ic_report


def plot_results(perf, benchmark, factor_cols, save_dir, ann_ret=0, sharpe=0):
    """绘制净值曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    save_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # 净值曲线
    ax1.plot(
        perf["date"], perf["cum_return"], color="#e53935", linewidth=2,
        label=f"多因子组合 (收益={ann_ret:+.1f}%, 夏普={sharpe:.2f})",
    )
    ax1.plot(
        benchmark["date"], benchmark["cum_return"], color="#607d8b", linewidth=1.5,
        linestyle="--", label="HS300等权基准",
    )
    ax1.axhline(y=1, color="#999", linewidth=0.5)
    ax1.set_title("多因子选股策略 — 净值曲线", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 回撤
    drawdown = perf["cum_return"] / perf["cum_return"].cummax() - 1
    ax2.fill_between(perf["date"], 0, drawdown * 100, color="#e53935", alpha=0.3)
    ax2.plot(perf["date"], drawdown * 100, color="#e53935", linewidth=1)
    ax2.set_title("回撤 (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "multi_factor_backtest.png", dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {save_dir / 'multi_factor_backtest.png'}")
    plt.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多因子选股回测")
    parser.add_argument("--plot", action="store_true", help="输出净值曲线图")
    parser.add_argument("--top", type=int, default=30, help="持仓股票数 (default: 30)")
    parser.add_argument("--pct", type=float, default=0, help="持仓比例 (default: 0=用top_n)")
    parser.add_argument("--min-ir", type=float, default=0.10, help="最小 IC_IR 阈值")
    parser.add_argument("--weighting", type=str, default="equal",
                        choices=["equal", "ic_ir"], help="因子加权方式")
    args = parser.parse_args()

    top_pct = args.pct if args.pct > 0 else 0.2
    top_n = args.top if args.top > 0 and args.pct == 0 else 0

    result, ic_report = run_backtest(
        start_date="2021-01-01",
        top_pct=top_pct,
        top_n=top_n,
        min_ic_ir=args.min_ir,
        weighting=args.weighting,
        plot=args.plot,
    )

    # 保存月度收益
    result["perf"].to_csv(ROOT / "data" / "backtest_monthly.csv",
                          index=False, encoding="utf-8-sig")
    print(f"\n月度收益已保存: data/backtest_monthly.csv")
