"""
滚动窗口样本外验证 (Walk-Forward Validation)

设计:
  训练 36 个月 → 测试 12 个月 → 滚动前进 12 个月
  每个窗口重新筛选因子、估计 IC、合成得分
  汇总全部 OOS 区间 → 与固定窗口 IS 结果对比

关键指标:
  - OOS 年化收益/夏普/回撤
  - IS-OOS 衰减比（衡量过拟合程度）
  - 因子选择稳定性（各窗口选中的因子一致性）
  - 逐年 OOS 表现（是否有某年特别差）

用法:
    python walkforward.py
    python walkforward.py --plot
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import build_clean_factor_table
from factor_test import ic_analysis
from multi_factor import select_factors, compute_composite_score


# ============================================================
# 1. 滚动窗口配置
# ============================================================

def generate_windows(dates: list, train_months: int = 36, test_months: int = 12,
                     step_months: int = 12) -> list:
    """
    生成滚动窗口

    返回: [(train_start, train_end, test_start, test_end), ...]
    """
    dates = sorted(dates)
    windows = []

    current_start = 0
    while True:
        train_start = current_start
        train_end = train_start + train_months
        test_start = train_end
        test_end = test_start + test_months

        if test_end > len(dates):
            # 最后一次测试窗口可以短一些
            if test_start >= len(dates) - 3:
                break
            test_end = len(dates)

        windows.append((
            dates[train_start], dates[train_end - 1],
            dates[test_start], dates[test_end - 1],
        ))
        current_start += step_months

        if current_start + train_months >= len(dates):
            break

    return windows


# ============================================================
# 2. 单窗口回测
# ============================================================

def run_one_window(factor_table: pd.DataFrame, train_start, train_end,
                   test_start, test_end, top_n: int = 30,
                   min_ic_ir: float = 0.10, min_t: float = 1.5) -> dict:
    """
    单个滚动窗口：训练选因子 → 测试回测

    返回: {test_perf_df, train_ic_report, selected_factors, train_metrics, test_metrics}
    """
    # 划分训练/测试
    train_df = factor_table[
        (factor_table["date"] >= train_start) &
        (factor_table["date"] <= train_end)
    ].copy()
    test_df = factor_table[
        (factor_table["date"] >= test_start) &
        (factor_table["date"] <= test_end)
    ].copy()

    if len(train_df) < 100 or len(test_df) < 10:
        return None

    # 训练期 IC 分析
    ic_report = ic_analysis(train_df)
    factor_cols = select_factors(train_df, ic_report,
                                 min_ic_ir=min_ic_ir, min_t=min_t,
                                 prefer_raw=True)

    if len(factor_cols) < 3:
        top5 = ic_report.head(5)["factor"].tolist()
        factor_cols = [f"factor_{f}" if not f.startswith("factor_") else f
                       for f in top5]
        factor_cols = [f for f in factor_cols if f in train_df.columns]

    # 方向检查（确保训练期 IC 方向正确）
    for col in factor_cols:
        ic_series = []
        for date, grp in train_df.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) >= 10:
                ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
                ic_series.append(ic)
        if len(ic_series) > 0 and np.mean(ic_series) < 0:
            train_df[col] = -train_df[col]
            test_df[col] = -test_df[col]

    # 在训练集上合成得分（用于估算权重）
    train_scored = compute_composite_score(train_df, factor_cols, weighting="equal")
    test_scored = compute_composite_score(test_df, factor_cols, weighting="equal")

    # 测试期回测
    prev = set()
    monthly_rets = []

    for date in sorted(test_scored["date"].unique()):
        month = test_scored[test_scored["date"] == date].dropna(
            subset=["composite_score", "forward_ret_1m"]
        )
        if len(month) < 10:
            continue

        month = month.sort_values("composite_score", ascending=False)
        # 按可用股票数动态调整选股数
        n_select = min(top_n, max(int(len(month) * 0.2), 10))
        selected = month.head(n_select)

        current = set(selected["code"].tolist())
        turnover = len(current - prev) / len(current) if prev else 1.0
        prev = current

        gross = selected["forward_ret_1m"].mean()
        net = gross - 0.003 * turnover * 2

        monthly_rets.append({
            "date": date, "return": net, "gross_return": gross,
            "turnover": turnover, "n_selected": n_select,
        })

    if len(monthly_rets) < 3:
        return None

    test_perf = pd.DataFrame(monthly_rets)
    test_perf["cum_return"] = (1 + test_perf["return"]).cumprod()

    # 训练期指标
    train_selected = train_scored.dropna(subset=["composite_score", "forward_ret_1m"])
    if len(train_selected) > 0:
        train_mean_ret = train_selected.groupby("date")["forward_ret_1m"].mean().mean()
        train_ann_ret = (1 + train_mean_ret) ** 12 - 1
    else:
        train_ann_ret = 0

    # 测试期指标
    rets = test_perf["return"].values
    test_ann_ret = (1 + np.mean(rets)) ** 12 - 1
    test_ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
    test_sharpe = (test_ann_ret - 0.02) / test_ann_vol if test_ann_vol > 0 else 0
    cum = test_perf["cum_return"].values
    test_max_dd = (cum / np.maximum.accumulate(cum) - 1).min()

    # IC 衰减（训练期前半 vs 后半）
    train_dates = train_df["date"].unique()
    mid = len(train_dates) // 2
    first_half_ic = {}
    second_half_ic = {}
    for col in factor_cols:
        fh = train_df[train_df["date"].isin(train_dates[:mid])]
        sh = train_df[train_df["date"].isin(train_dates[mid:])]
        for label, subset in [("first", fh), ("second", sh)]:
            ic_list = []
            for date, grp in subset.groupby("date"):
                valid = grp[[col, "forward_ret_1m"]].dropna()
                if len(valid) >= 10:
                    ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
                    ic_list.append(ic)
            avg = np.mean(ic_list) if ic_list else 0
            if label == "first":
                first_half_ic[col] = avg
            else:
                second_half_ic[col] = avg

    ic_decayed = 0
    for col in factor_cols:
        if col in first_half_ic and col in second_half_ic:
            if abs(second_half_ic[col]) < abs(first_half_ic[col]) * 0.7:
                ic_decayed += 1

    return {
        "test_perf": test_perf,
        "test_start": test_start,
        "test_end": test_end,
        "train_start": train_start,
        "train_end": train_end,
        "test_ann_ret": round(test_ann_ret * 100, 2),
        "test_sharpe": round(test_sharpe, 2),
        "test_max_dd": round(test_max_dd * 100, 2),
        "train_ann_ret": round(train_ann_ret * 100, 2),
        "selected_factors": [c.replace("factor_", "") for c in factor_cols],
        "n_factors": len(factor_cols),
        "n_months": len(test_perf),
        "ic_decayed_factors": ic_decayed,
        "avg_turnover": round(test_perf["turnover"].mean(), 3),
    }


# ============================================================
# 3. 全流程滚动验证
# ============================================================

def run_walkforward(
    train_months: int = 36,
    test_months: int = 12,
    step_months: int = 12,
    top_n: int = 30,
    plot: bool = False,
):
    """滚动窗口验证主流程"""
    print("=" * 60)
    print("滚动窗口样本外验证 (Walk-Forward)")
    print(f"训练={train_months}月, 测试={test_months}月, 步长={step_months}月")
    print("=" * 60)

    # 1. 构建完整因子表
    print("\n[1/3] 构建因子表...")
    # 只用 raw 因子，不做中性化（之前已验证 raw 更有效）
    factor_table = build_clean_factor_table(ROOT, start_date="2021-01-01",
                                            neutralize=False)
    dates = sorted(factor_table["date"].unique())
    print(f"  数据: {len(dates)} 个月, {factor_table['code'].nunique()} 只股票")

    # 2. 生成窗口
    windows = generate_windows(dates, train_months, test_months, step_months)
    print(f"\n  窗口数: {len(windows)}")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"  W{i + 1}: 训练 [{str(tr_s.date())} → {str(tr_e.date())}]  "
              f"测试 [{str(te_s.date())} → {str(te_e.date())}]")

    # 3. 逐窗口回测
    print(f"\n[2/3] 滚动回测...")
    window_results = []
    all_oos_rets = []
    factor_occurrence = {}

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(tqdm(windows, desc="滚动窗口")):
        wr = run_one_window(factor_table, tr_s, tr_e, te_s, te_e, top_n=top_n)
        if wr is None:
            print(f"  W{i + 1}: 数据不足，跳过")
            continue

        window_results.append(wr)
        all_oos_rets.append(wr["test_perf"][["date", "return"]])

        for f in wr["selected_factors"]:
            factor_occurrence[f] = factor_occurrence.get(f, 0) + 1

        print(f"  W{i + 1}: OOS {wr['test_ann_ret']:+.1f}% | "
              f"夏普 {wr['test_sharpe']:.2f} | "
              f"回撤 {wr['test_max_dd']:.1f}% | "
              f"因子 {wr['n_factors']}个: {wr['selected_factors'][:5]}...")

    if not window_results:
        print("没有足够的窗口进行验证")
        return None

    # 4. 汇总分析
    print(f"\n[3/3] 汇总分析")
    print("=" * 60)

    # 拼接全部 OOS 收益
    oos_all = pd.concat(all_oos_rets).sort_values("date")
    oos_all["cum_return"] = (1 + oos_all["return"]).cumprod()

    oos_rets = oos_all["return"].values
    oos_ann_ret = (1 + np.mean(oos_rets)) ** 12 - 1
    oos_ann_vol = np.std(oos_rets, ddof=1) * np.sqrt(12)
    oos_sharpe = (oos_ann_ret - 0.02) / oos_ann_vol if oos_ann_vol > 0 else 0
    oos_cum = oos_all["cum_return"].values
    oos_max_dd = (oos_cum / np.maximum.accumulate(oos_cum) - 1).min()
    oos_win_rate = (oos_rets > 0).mean()

    # IS 基准（全样本回测）
    from multi_factor import backtest_portfolio
    ic_report_full = ic_analysis(factor_table)
    factor_cols_full = select_factors(factor_table, ic_report_full,
                                      min_ic_ir=0.10, min_t=1.5, prefer_raw=True)
    scored_full = compute_composite_score(factor_table, factor_cols_full,
                                          weighting="equal")
    is_result = backtest_portfolio(scored_full, top_n=top_n)

    # IS-OOS 衰减
    is_ann_ret = is_result["ann_return"] / 100  # 转小数
    oos_is_ratio = (1 + oos_ann_ret) / (1 + is_ann_ret) - 1 if is_ann_ret > -1 else np.nan

    print(f"\n{'=' * 50}")
    print(f"IS vs OOS 对比")
    print(f"{'=' * 50}")
    print(f"{'指标':<16} {'IS(全样本)':>12} {'OOS(滚动)':>12} {'衰减':>10}")
    print(f"{'-' * 50}")
    print(f"{'年化收益':<16} {is_result['ann_return']:>+10.1f}%  "
          f"{oos_ann_ret*100:>+10.1f}%  {oos_is_ratio*100:>+9.1f}%")
    print(f"{'夏普比率':<16} {is_result['sharpe']:>12.2f}  "
          f"{oos_sharpe:>12.2f}")
    print(f"{'最大回撤':<16} {is_result['max_dd']:>+10.1f}%  "
          f"{oos_max_dd*100:>+10.1f}%")
    print(f"{'胜率':<16} {is_result['win_rate']*100:>11.0f}%  "
          f"{oos_win_rate*100:>11.0f}%")

    # 逐年表现
    print(f"\n{'=' * 50}")
    print(f"逐年 OOS 表现")
    print(f"{'=' * 50}")
    oos_all["year"] = oos_all["date"].dt.year
    yearly = oos_all.groupby("year").apply(
        lambda g: pd.Series({
            "月份": len(g),
            "累计收益": (g["return"] + 1).prod() - 1,
            "年化": (1 + g["return"].mean()) ** 12 - 1,
            "夏普": (g["return"].mean() / g["return"].std() * np.sqrt(12))
            if g["return"].std() > 0 else 0,
            "胜率": (g["return"] > 0).mean(),
        })
    ).reset_index()
    for _, row in yearly.iterrows():
        print(f"  {int(row['year'])}年: {row['年化']*100:+.1f}% | "
              f"夏普 {row['夏普']:.2f} | "
              f"胜率 {row['胜率']*100:.0f}% | "
              f"{int(row['月份'])}个月")

    # 因子稳定性
    print(f"\n{'=' * 50}")
    print(f"因子选择稳定性")
    print(f"{'=' * 50}")
    n_windows = len(window_results)
    stable_factors = {k: v for k, v in sorted(
        factor_occurrence.items(), key=lambda x: -x[1]
    ) if v >= n_windows * 0.5}  # 至少一半窗口选中
    unstable_factors = {k: v for k, v in factor_occurrence.items()
                        if v < n_windows * 0.5}

    print(f"  稳定因子 (>=50%窗口, {len(stable_factors)}个):")
    for f, count in stable_factors.items():
        bar = "#" * count + "-" * (n_windows - count)
        print(f"    {f:24s} {bar} {count}/{n_windows}")

    if unstable_factors:
        print(f"\n  不稳定因子 (<50%窗口, {len(unstable_factors)}个):")
        for f, count in unstable_factors.items():
            print(f"    {f:24s} {count}/{n_windows}")

    # 过拟合风险评级
    if oos_sharpe >= is_result["sharpe"] * 0.7:
        overfit_level = "低"
    elif oos_sharpe >= is_result["sharpe"] * 0.4:
        overfit_level = "中"
    else:
        overfit_level = "高"

    print(f"\n{'=' * 50}")
    print(f"过拟合风险评级: {overfit_level}")
    print(f"  IS 夏普 {is_result['sharpe']:.2f} → OOS 夏普 {oos_sharpe:.2f}")
    print(f"  OOS/IS 保留率: {oos_sharpe/is_result['sharpe']*100:.0f}%"
          if is_result['sharpe'] > 0 else "  N/A")
    print(f"{'=' * 50}")

    # 保存
    save_dir = ROOT / "data"
    oos_all[["date", "return", "cum_return"]].to_csv(
        save_dir / "walkforward_oos.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(window_results).drop(columns=["test_perf"]).to_csv(
        save_dir / "walkforward_windows.csv", index=False, encoding="utf-8-sig")
    print(f"\n数据已保存: data/walkforward_*.csv")

    # 图表
    if plot:
        plot_walkforward(oos_all, is_result, window_results,
                         save_dir / "plots")

    return {
        "oos_ann_ret": round(oos_ann_ret * 100, 2),
        "oos_sharpe": round(oos_sharpe, 2),
        "oos_max_dd": round(oos_max_dd * 100, 2),
        "is_ann_ret": is_result["ann_return"],
        "is_sharpe": is_result["sharpe"],
        "overfit_level": overfit_level,
        "windows": window_results,
    }


# ============================================================
# 4. 可视化
# ============================================================

def plot_walkforward(oos_all, is_result, window_results, save_dir):
    """滚动验证全景图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 12))

    # --- 图1: IS vs OOS 净值曲线 ---
    ax1 = fig.add_subplot(2, 3, 1)
    # IS
    is_perf = is_result["perf"]
    ax1.plot(is_perf["date"], is_perf["cum_return"],
             color="#607d8b", linewidth=1.5, linestyle="--",
             label=f"IS全样本 (夏普={is_result['sharpe']:.2f})")
    # OOS
    ax1.plot(oos_all["date"], oos_all["cum_return"],
             color="#e53935", linewidth=2,
             label="OOS滚动验证")
    # 窗口分隔线
    for wr in window_results:
        ax1.axvline(x=wr["test_start"], color="#ccc", linewidth=0.5, alpha=0.5)
    ax1.axhline(y=1, color="#999", linewidth=0.5)
    ax1.set_title("IS vs OOS 净值曲线", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # --- 图2: 逐年 OOS 表现 ---
    ax2 = fig.add_subplot(2, 3, 2)
    oos_all = oos_all.copy()
    oos_all["year"] = oos_all["date"].dt.year
    yearly = oos_all.groupby("year")["return"].apply(
        lambda g: (1 + g).prod() - 1
    )
    colors = ["#4caf50" if v > 0 else "#f44336" for v in yearly.values]
    ax2.bar(yearly.index.astype(str), yearly.values * 100, color=colors)
    ax2.axhline(y=0, color="#333", linewidth=0.5)
    ax2.set_title("逐年 OOS 收益", fontsize=12, fontweight="bold")
    ax2.set_ylabel("累计收益 %")
    ax2.grid(True, alpha=0.2, axis="y")

    # --- 图3: 窗口间因子选择一致性 ---
    ax3 = fig.add_subplot(2, 3, 3)
    from collections import Counter
    factor_counts = Counter()
    for wr in window_results:
        factor_counts.update(wr["selected_factors"])
    top = factor_counts.most_common(10)
    if top:
        names, counts = zip(*top)
        ax3.barh(range(len(names)), counts, color="#2196f3")
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=8)
        ax3.set_title(f"因子选中频率 ({len(window_results)}个窗口)",
                      fontsize=12, fontweight="bold")
        ax3.set_xlabel("选中次数")

    # --- 图4: 窗口间 IC 衰减 ---
    ax4 = fig.add_subplot(2, 3, 4)
    window_labels = [f"W{i + 1}" for i in range(len(window_results))]
    n_factors = [wr["n_factors"] for wr in window_results]
    decayed = [wr["ic_decayed_factors"] for wr in window_results]
    x = range(len(window_results))
    ax4.bar(x, n_factors, color="#2196f3", label="选中因子数")
    ax4.bar(x, decayed, color="#ff9800", label="IC衰减因子")
    ax4.set_xticks(x)
    ax4.set_xticklabels(window_labels)
    ax4.set_title("各窗口因子衰减情况", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2, axis="y")

    # --- 图5: OOS 收益月度分布 ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(oos_all["return"] * 100, bins=30, color="#e53935",
             alpha=0.7, edgecolor="white")
    ax5.axvline(x=0, color="#333", linewidth=0.5, linestyle="--")
    ax5.set_title(f"OOS 月收益分布 (均值={np.mean(oos_all['return'])*100:.1f}%)",
                  fontsize=12, fontweight="bold")
    ax5.set_xlabel("月收益 %")
    ax5.grid(True, alpha=0.2, axis="y")

    # --- 图6: 综合评估表 ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    oos_rets = oos_all["return"].values
    oos_ann_ret = (1 + np.mean(oos_rets)) ** 12 - 1
    oos_ann_vol = np.std(oos_rets, ddof=1) * np.sqrt(12)
    oos_sharpe = (oos_ann_ret - 0.02) / oos_ann_vol if oos_ann_vol > 0 else 0
    oos_cum = oos_all["cum_return"].values
    oos_max_dd = (oos_cum / np.maximum.accumulate(oos_cum) - 1).min()
    oos_win = (oos_rets > 0).mean()

    table_data = [
        ["年化收益", f"{is_result['ann_return']:+.1f}%",
         f"{oos_ann_ret*100:+.1f}%"],
        ["夏普比率", f"{is_result['sharpe']:.2f}",
         f"{oos_sharpe:.2f}"],
        ["最大回撤", f"{is_result['max_dd']:.1f}%",
         f"{oos_max_dd*100:.1f}%"],
        ["胜率", f"{is_result['win_rate']*100:.0f}%",
         f"{oos_win*100:.0f}%"],
        ["总月份", f"{len(is_perf)}",
         f"{len(oos_all)}"],
    ]

    table = ax6.table(
        cellText=table_data,
        colLabels=["指标", "IS (全样本)", "OOS (滚动)"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    for j in range(3):
        table[0, j].set_facecolor("#e53935")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax6.set_title("IS vs OOS 综合对比", fontsize=12, fontweight="bold",
                  y=1.02)

    plt.tight_layout()
    plt.savefig(save_dir / "walkforward.png", dpi=150, bbox_inches="tight")
    print(f"图表已保存: {save_dir / 'walkforward.png'}")
    plt.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="滚动窗口验证")
    parser.add_argument("--plot", action="store_true", help="输出图表")
    parser.add_argument("--top", type=int, default=30, help="持仓数")
    parser.add_argument("--train", type=int, default=36, help="训练月数")
    parser.add_argument("--test", type=int, default=12, help="测试月数")
    parser.add_argument("--step", type=int, default=12, help="步长月数")
    args = parser.parse_args()

    run_walkforward(
        train_months=args.train,
        test_months=args.test,
        step_months=args.step,
        top_n=args.top,
        plot=args.plot,
    )
