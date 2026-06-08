"""
IC 衰减跟踪 & 因子健康诊断

功能:
  - 滚动 IC（12月窗口）：识别因子近期表现
  - IC 衰减检验：回归 IC ~ 时间，负斜率 = 衰减信号
  - IC 热力图：因子 × 年份矩阵
  - 因子健康评分：综合近期 IC、衰减趋势、稳定性
  - 动态权重生成：用于多因子组合的下一期权重

用法:
    python ic_decay.py
    python ic_decay.py --plot
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
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 截面 IC 时间序列提取
# ============================================================

def extract_ic_series(factor_table: pd.DataFrame) -> pd.DataFrame:
    """
    提取每个因子每期的截面 Rank IC

    返回: DataFrame [date, factor1, factor2, ...], 每格为该期 IC 值
    """
    factor_cols = [c for c in factor_table.columns if c.startswith("factor_")]
    dates = sorted(factor_table["date"].unique())

    ic_data = []
    for date in tqdm(dates, desc="提取 IC"):
        row = {"date": date}
        cross = factor_table[factor_table["date"] == date]
        for col in factor_cols:
            valid = cross[[col, "forward_ret_1m"]].dropna()
            if len(valid) >= 10:
                ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
                row[col.replace("factor_", "")] = ic
            else:
                row[col.replace("factor_", "")] = np.nan
        ic_data.append(row)

    return pd.DataFrame(ic_data).set_index("date")


# ============================================================
# 2. 滚动 IC
# ============================================================

def rolling_ic(ic_df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    滚动 IC 均值（默认 12 个月）

    返回: DataFrame，值 = 过去 window 个月的平均 IC
    """
    return ic_df.rolling(window, min_periods=6).mean()


# ============================================================
# 3. IC 衰减检验
# ============================================================

def ic_decay_test(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个因子做 IC 衰减检验

    H0: IC 不随时间衰减
    方法: IC_t = α + β × t + ε  → 检验 β 是否显著 < 0

    返回: DataFrame
        factor, ic_mean, recent_ic (近12月), ic_trend (β), trend_t, decay_severity
    """
    results = []
    t = np.arange(len(ic_df))

    for col in ic_df.columns:
        series = ic_df[col].dropna()
        if len(series) < 24:
            continue

        # 整体统计
        ic_mean = series.mean()
        ic_std = series.std(ddof=1)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0

        # 近 12 月
        recent = series.iloc[-12:].mean() if len(series) >= 12 else ic_mean

        # 衰减回归
        common_t = t[-len(series):]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            common_t, series.values
        )

        # 衰减严重度
        # 负斜率 + 显著 + 近期低于历史 = 严重衰减
        decay_score = 0
        if slope < 0:
            decay_score += 2  # 负趋势
        if p_value < 0.1:
            decay_score += 2  # 统计显著
        if recent < ic_mean:
            decay_score += 1  # 近期低于均值

        results.append({
            "factor": col,
            "ic_mean": round(ic_mean, 4),
            "ic_ir": round(ic_ir, 4),
            "recent_ic": round(recent, 4),
            "ic_change": round(recent - ic_mean, 4),
            "trend_beta": round(slope, 6),
            "trend_t": round(slope / std_err, 3) if std_err > 0 else 0,
            "trend_p": round(p_value, 4),
            "decay_score": decay_score,
            "n_months": len(series),
        })

    report = pd.DataFrame(results).sort_values("decay_score", ascending=False)
    return report


# ============================================================
# 4. IC 热力图（年份 × 因子）
# ============================================================

def ic_heatmap_data(ic_df: pd.DataFrame) -> pd.DataFrame:
    """按年份汇总 IC 均值，用于热力图"""
    ic_df = ic_df.copy()
    ic_df["year"] = ic_df.index.year
    yearly_ic = ic_df.groupby("year").mean()

    # 只保留有效因子
    valid_cols = yearly_ic.dropna(axis=1, how="all").columns
    return yearly_ic[valid_cols]


# ============================================================
# 5. 因子健康评分
# ============================================================

def factor_health_score(decay_report: pd.DataFrame) -> pd.DataFrame:
    """
    综合评分: 0-10 分

    评分构成:
      - IC 水平 (0-4分):    |IC_IR| 越高越好
      - 衰减趋势 (0-3分):   无衰减=3, 轻微=2, 显著=1, 严重=0
      - 稳定性 (0-3分):     近期与历史一致性

    返回: 健康报告 DataFrame
    """
    df = decay_report.copy()

    # IC 水平得分
    max_ir = abs(df["ic_ir"]).max()
    if max_ir > 0:
        df["score_ic"] = (abs(df["ic_ir"]) / max_ir * 4).clip(0, 4)

    # 衰减得分: decay_score 0=健康, 5=严重
    df["score_decay"] = (3 - df["decay_score"].clip(0, 3))

    # 稳定性得分: |recent - mean| 越小越好
    ic_std = df["ic_change"].abs().std()
    if ic_std > 0:
        df["score_stability"] = (
            3 - (df["ic_change"].abs() / (ic_std * 2)).clip(0, 3)
        )

    df["health_score"] = (
        df["score_ic"] + df["score_decay"] + df["score_stability"]
    ).round(1)

    return df.sort_values("health_score", ascending=False)


# ============================================================
# 6. 动态权重生成
# ============================================================

def generate_dynamic_weights(health_report: pd.DataFrame,
                             min_score: float = 4.0) -> dict:
    """
    基于健康评分生成下一期因子权重

    规则:
      - health_score < min_score: 剔除（权重=0）
      - 其余按 health_score 加权
    """
    valid = health_report[health_report["health_score"] >= min_score].copy()
    if len(valid) == 0:
        valid = health_report.head(5).copy()  # 兜底

    valid["weight"] = valid["health_score"] / valid["health_score"].sum()

    print(f"\n动态权重 (health >= {min_score}): {len(valid)}/{len(health_report)} 个因子入选")
    for _, row in valid.iterrows():
        print(f"  {row['factor']:20s}: health={row['health_score']:.1f}, "
              f"weight={row['weight']:.3f}")

    return dict(zip("factor_" + valid["factor"], valid["weight"]))


# ============================================================
# 7. 可视化
# ============================================================

def plot_decay_dashboard(ic_df, decay_report, rolling_ic_df, save_dir):
    """IC 衰减全景仪表盘"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    save_dir.mkdir(parents=True, exist_ok=True)

    # 选排名前 8 的因子
    top_factors = decay_report.head(8)["factor"].tolist()
    available = [f for f in top_factors if f in ic_df.columns]
    n = len(available)

    fig = plt.figure(figsize=(18, 12))

    # --- 左上: IC 累计曲线 ---
    ax1 = fig.add_subplot(2, 3, 1)
    colors = plt.cm.tab10(range(n))
    for i, f in enumerate(available):
        cum_ic = ic_df[f].dropna().cumsum()
        ax1.plot(cum_ic.index, cum_ic.values, color=colors[i],
                 linewidth=1.5, label=f, alpha=0.85)
    ax1.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax1.set_title("IC 累计曲线", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7, loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.2)

    # --- 中上: 滚动 12M IC ---
    ax2 = fig.add_subplot(2, 3, 2)
    for i, f in enumerate(available):
        if f in rolling_ic_df.columns:
            ax2.plot(rolling_ic_df.index, rolling_ic_df[f].values,
                     color=colors[i], linewidth=1.5, alpha=0.85)
    ax2.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax2.set_title("滚动 12 月 IC 均值", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    # --- 右上: 衰减趋势（近 3 年逐个因子） ---
    ax3 = fig.add_subplot(2, 3, 3)
    recent_3y = ic_df.iloc[-36:]
    x = range(len(recent_3y))
    for i, f in enumerate(available[:5]):
        s = recent_3y[f].dropna()
        if len(s) < 6:
            continue
        # 散点 + 趋势线
        idx = [recent_3y.index.get_loc(d) for d in s.index]
        ax3.scatter(idx, s.values, color=colors[i], s=12, alpha=0.5)
        slope, intercept, _, _, _ = stats.linregress(idx, s.values)
        ax3.plot(idx, intercept + slope * np.array(idx),
                 color=colors[i], linewidth=2, label=f"{f} (β={slope*100:.2f}/月)")
    ax3.axhline(y=0, color="#999", linewidth=0.5, linestyle="--")
    ax3.set_title("近 3 年 IC 衰减趋势", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=7, loc="best")
    ax3.grid(True, alpha=0.2)

    # --- 左下: 因子健康度条形图 ---
    ax4 = fig.add_subplot(2, 3, 4)
    health_scores = decay_report.set_index("factor")["decay_score"].sort_values()
    bars = ax4.barh(range(len(health_scores)), health_scores.values,
                     color=["#4caf50" if v <= 1 else "#ff9800" if v <= 3 else "#f44336"
                            for v in health_scores.values])
    ax4.set_yticks(range(len(health_scores)))
    ax4.set_yticklabels(health_scores.index, fontsize=8)
    ax4.set_title("IC 衰减严重度 (越低越健康)", fontsize=12, fontweight="bold")
    ax4.axvline(x=2, color="#999", linewidth=0.5, linestyle="--")
    for i, (_, v) in enumerate(health_scores.items()):
        ax4.text(v + 0.05, i, str(v), va="center", fontsize=8)

    # --- 中下: IC 热力图 ---
    ax5 = fig.add_subplot(2, 3, 5)
    heatmap = ic_heatmap_data(ic_df)
    top_cols = available[:8]
    hm_data = heatmap[[c for c in top_cols if c in heatmap.columns]]
    im = ax5.imshow(hm_data.T.values, aspect="auto", cmap="RdYlGn",
                     vmin=-0.1, vmax=0.1, interpolation="none")
    ax5.set_xticks(range(len(hm_data.index)))
    ax5.set_xticklabels([str(y) for y in hm_data.index], fontsize=8)
    ax5.set_yticks(range(len(hm_data.columns)))
    ax5.set_yticklabels(hm_data.columns, fontsize=8)
    ax5.set_title("IC 热力图 (年份 × 因子)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # --- 右下: 健康评分表 ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    health = factor_health_score(decay_report)
    table_data = health.head(10)[["factor", "health_score", "ic_ir", "recent_ic",
                                   "decay_score"]].copy()
    table_data.columns = ["因子", "健康分", "IC_IR", "近期IC", "衰减度"]

    table = ax6.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # 表头样式
    for j in range(len(table_data.columns)):
        table[0, j].set_facecolor("#e53935")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax6.set_title("因子健康评分 Top 10", fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_dir / "ic_decay_dashboard.png", dpi=150, bbox_inches="tight")
    print(f"\n仪表盘已保存: {save_dir / 'ic_decay_dashboard.png'}")
    plt.close()


# ============================================================
# 8. 主流程
# ============================================================

def run_ic_decay_analysis(plot: bool = False):
    """全流程 IC 衰减分析"""
    print("=" * 60)
    print("IC 衰减跟踪 & 因子健康诊断")
    print("=" * 60)

    # 1. 构建因子表
    print("\n[1/5] 构建因子表...")
    df = build_clean_factor_table(ROOT, start_date="2021-01-01", neutralize=False)

    # 2. 提取 IC 序列
    print("\n[2/5] 提取截面 IC...")
    ic_df = extract_ic_series(df)
    print(f"  IC 矩阵: {ic_df.shape[0]} 期 × {ic_df.shape[1]} 因子")

    # 3. 滚动 IC
    print("\n[3/5] 计算滚动 IC...")
    roll_ic = rolling_ic(ic_df, window=12)

    # 4. 衰减检验
    print("\n[4/5] IC 衰减检验...")
    decay = ic_decay_test(ic_df)

    # 分类
    severe = decay[decay["decay_score"] >= 3]
    mild = decay[(decay["decay_score"] >= 1) & (decay["decay_score"] <= 2)]
    healthy = decay[decay["decay_score"] == 0]

    print(f"\n{'=' * 50}")
    print(f"衰减诊断结果")
    print(f"{'=' * 50}")
    print(f"  严重衰减 (score>=3): {len(severe)} 个")
    print(f"  轻微衰减 (1-2):      {len(mild)} 个")
    print(f"  健康 (0):            {len(healthy)} 个")

    if len(severe) > 0:
        print(f"\n  !! 严重衰减因子:")
        for _, r in severe.iterrows():
            print(f"    {r['factor']:24s}  IC={r['ic_mean']:+.4f}  "
                  f"近期={r['recent_ic']:+.4f}  β={r['trend_beta']*100:+.3f}%/月  "
                  f"t={r['trend_t']:+.1f}")

    if len(healthy) > 0:
        print(f"\n  ** 健康因子 (Top 8):")
        for _, r in healthy.head(8).iterrows():
            print(f"    {r['factor']:24s}  IC={r['ic_mean']:+.4f}  "
                  f"IR={r['ic_ir']:+.3f}  近期={r['recent_ic']:+.4f}")

    # 5. 健康评分 + 动态权重
    print("\n[5/5] 因子健康评分 & 动态权重...")
    health = factor_health_score(decay)
    weights = generate_dynamic_weights(health, min_score=4.0)

    # 保存
    save_dir = ROOT / "data"
    ic_df.to_csv(save_dir / "ic_series.csv", encoding="utf-8-sig")
    decay.to_csv(save_dir / "ic_decay_report.csv", index=False, encoding="utf-8-sig")
    health.to_csv(save_dir / "factor_health.csv", index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: data/ic_series.csv, ic_decay_report.csv, factor_health.csv")

    # 图表
    if plot:
        plot_decay_dashboard(ic_df, decay, roll_ic, save_dir / "plots")

    return ic_df, decay, health, weights


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IC 衰减跟踪")
    parser.add_argument("--plot", action="store_true", help="输出仪表盘图")
    args = parser.parse_args()

    ic_df, decay, health, weights = run_ic_decay_analysis(plot=args.plot)
