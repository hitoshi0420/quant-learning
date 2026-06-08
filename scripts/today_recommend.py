"""
今日开盘推荐 — 基于最新月频因子数据，严格无未来信息
用法: python today_recommend.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import compute_daily_factors, _align_financials, load_all

# ---- 加载股票名称映射 ----
name_df = pd.read_csv(ROOT / "data" / "stock_names.csv", dtype={"code_clean": str})
name_map = dict(zip(name_df["code_clean"], name_df["name"]))

def ic_analysis_train(train_df):
    factor_cols = [c for c in train_df.columns if c.startswith("factor_")]
    results = []
    for col in factor_cols:
        ic_list = []
        for date, grp in train_df.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) >= 10:
                try:
                    ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
                    ic_list.append(ic)
                except Exception as e:
                    print(f"[today_recommend] spearmanr {col} 失败: {e}")
        if len(ic_list) >= 6:
            ic_arr = np.array(ic_list)
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr, ddof=1)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
            results.append({"factor": col, "ic_mean": ic_mean, "ic_ir": ic_ir, "t_stat": t_stat})
    return pd.DataFrame(results).sort_values("ic_ir", key=abs, ascending=False)


def estimate_covariance(ret_matrix, shrinkage=0.2):
    n = ret_matrix.shape[1]
    sample_cov = np.cov(ret_matrix, rowvar=False)
    diag_vals = np.diag(sample_cov)
    sqrt_d = np.sqrt(np.maximum(diag_vals, 0))
    target_cov = np.outer(sqrt_d, sqrt_d)
    np.fill_diagonal(target_cov, diag_vals)
    return (1 - shrinkage) * sample_cov + shrinkage * target_cov


def max_sharpe_weights(mu, cov, max_w=0.10, min_w=0.01):
    n = len(mu)

    def neg_sharpe(w):
        pr = np.dot(w, mu)
        pv = np.sqrt(max(np.dot(w, np.dot(cov, w)), 1e-10))
        return -pr / pv

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w) for _ in range(n)]
    x0 = np.ones(n) / n
    result = minimize(neg_sharpe, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-8})
    if result.success:
        w = np.maximum(result.x, 0)
        s = w.sum()
        return w / s if s > 0 else x0
    return x0


def build_factor_table_with_price(start_date="2020-01-01"):
    daily, fin, _ = load_all(ROOT)
    df = compute_daily_factors(daily, fin)
    fin_daily = _align_financials(df[["date"]], fin)

    fin_map = {
        "roe_avg": "factor_roe", "np_margin": "factor_np_margin",
        "gp_margin": "factor_gp_margin", "yoy_ni": "factor_yoy_ni",
        "yoy_equity": "factor_yoy_equity",
        "current_ratio": "factor_current_ratio",
        "quick_ratio": "factor_quick_ratio",
        "liability_to_asset": "factor_liability_to_asset",
        "asset_turn_ratio": "factor_asset_turn",
        "cfo_to_np": "factor_cfo_to_np",
    }
    available_fin = {k: v for k, v in fin_map.items() if k in fin_daily.columns}
    for src, dst in available_fin.items():
        fin_daily[dst] = fin_daily[src]
        if src == "liability_to_asset":
            fin_daily[dst] = -fin_daily[src]

    df = df.merge(
        fin_daily[["date", "code"] + list(available_fin.values())],
        on=["date", "code"], how="left",
    )
    for col in available_fin.values():
        df[col] = df.groupby("code")[col].transform(lambda x: x.ffill())

    df["year_month"] = df["date"].dt.to_period("M")
    monthly_dates = df.groupby("year_month")["date"].max().reset_index(drop=True)
    monthly_dates = sorted(monthly_dates.tolist())

    month_end = df[df["date"].isin(monthly_dates)].copy()
    month_end = month_end.sort_values(["code", "date"])
    month_end["forward_close"] = month_end.groupby("code")["close"].shift(-1)
    month_end["forward_ret_1m"] = month_end["forward_close"] / month_end["close"] - 1
    month_end = month_end[month_end["date"] >= pd.Timestamp(start_date)]

    factor_cols = [c for c in month_end.columns if c.startswith("factor_")]
    out_cols = ["date", "code", "close"] + factor_cols + ["forward_ret_1m"]
    result = month_end[out_cols].dropna(subset=["forward_ret_1m"]).copy()
    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    for col in tqdm(factor_cols, desc="标准化因子"):
        result[col] = result.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    return result


def main():
    print("=" * 70)
    print("  今日开盘推荐 — 多因子量化策略 (严格无未来信息)")
    print("=" * 70)

    # 1. 构建因子表
    print("\n[1/5] 构建因子表...")
    factor_table = build_factor_table_with_price(start_date="2020-01-01")
    factor_cols_all = [c for c in factor_table.columns if c.startswith("factor_")]
    all_dates = sorted(factor_table["date"].unique())
    latest_date = all_dates[-1]
    print(f"  最新月频数据: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  因子表覆盖: {factor_table['date'].nunique()} 个月, {factor_table['code'].nunique()} 只股票")

    # 2. 因子训练 (只用最新日期之前的数据)
    print(f"\n[2/5] IC分析 (训练数据截止 {latest_date.strftime('%Y-%m-%d')} 之前)...")
    train_data = factor_table[factor_table["date"] < latest_date].copy()
    ic_report = ic_analysis_train(train_data)

    # 显示前10因子
    print(f"\n  {'因子':25s} {'IC均值':>8s} {'IC_IR':>8s} {'T值':>8s}")
    print(f"  {'-' * 50}")
    for _, row in ic_report.head(15).iterrows():
        name = row["factor"].replace("factor_", "").replace("_raw", "")
        print(f"  {name:25s} {row['ic_mean']:>+7.4f} {row['ic_ir']:>+7.3f} {row['t_stat']:>+7.2f}")

    # 3. 筛选显著因子
    significant = ic_report[
        (abs(ic_report["ic_ir"]) > 0.08) |
        (abs(ic_report["t_stat"]) > 1.2)
    ]
    if len(significant) < 5:
        significant = ic_report.head(8)

    # 去重选 raw 版本
    selected_factors = []
    seen = set()
    for _, row in significant.iterrows():
        f = row["factor"]
        base = f.replace("_raw", "")
        if base in seen:
            continue
        seen.add(base)
        raw_name = f"{base}_raw"
        if raw_name in factor_cols_all and base in factor_cols_all:
            chosen = raw_name
        elif raw_name in factor_cols_all:
            chosen = raw_name
        else:
            chosen = f
        if chosen in factor_cols_all:
            selected_factors.append(chosen)

    if len(selected_factors) < 3:
        selected_factors = [ic_report.iloc[j]["factor"] for j in range(min(5, len(ic_report)))]
        selected_factors = [f for f in selected_factors if f in factor_cols_all]

    ic_signs = {}
    for _, row in ic_report.iterrows():
        ic_signs[row["factor"]] = 1 if row["ic_mean"] >= 0 else -1

    print(f"\n[3/5] 入选因子 ({len(selected_factors)} 个):")
    for f in selected_factors:
        name = f.replace("factor_", "").replace("_raw", "")
        print(f"  {name:30s} IC_IR={ic_report[ic_report['factor']==f]['ic_ir'].values[0]:+.3f}")

    # 4. 选股打分
    print(f"\n[4/5] 最新月份 ({latest_date.strftime('%Y-%m-%d')}) 选股打分...")
    current_month = factor_table[factor_table["date"] == latest_date].copy()
    score = np.zeros(len(current_month))
    count = 0
    for col in selected_factors:
        if col in current_month.columns:
            vals = current_month[col].fillna(0).values
            if ic_signs.get(col, 1) < 0:
                vals = -vals
            score += vals
            count += 1
    current_month["composite_score"] = score / max(count, 1)
    current_month = current_month.sort_values("composite_score", ascending=False)

    top_n = 30
    selected = current_month.head(top_n)
    selected_codes = selected["code"].tolist()

    # 拉取历史收益做协方差估计
    past_dates = sorted(train_data["date"].unique()[-12:])
    past_returns = train_data[train_data["date"].isin(past_dates)]

    ret_data = {}
    prices = {}
    for code in selected_codes:
        stock_rets = past_returns[past_returns["code"] == code][["date", "forward_ret_1m"]]
        ret_data[code] = stock_rets.set_index("date")["forward_ret_1m"]
        price_row = selected[selected["code"] == code]
        if len(price_row) > 0:
            prices[code] = price_row["close"].iloc[0]

    ret_df = pd.DataFrame(ret_data).dropna(axis=1)
    common_codes = [c for c in selected_codes if c in ret_df.columns]

    # 5. 权重优化
    print(f"\n[5/5] Max Sharpe 权重优化 ({len(common_codes)} 只有效收益历史)...")
    if len(common_codes) >= 5:
        cov = estimate_covariance(ret_df[common_codes].values)
        mu = ret_df[common_codes].mean().values
        weights = max_sharpe_weights(mu, cov)
    else:
        common_codes = selected_codes[:min(len(selected_codes), top_n)]
        weights = np.ones(len(common_codes)) / len(common_codes)

    weights = weights / weights.sum()

    # ================================================================
    # 输出推荐组合
    # ================================================================
    total_capital = 10000.0

    print(f"\n{'=' * 70}")
    print(f"  今日推荐组合 (数据截止 {latest_date.strftime('%Y-%m-%d')})")
    print(f"  基准本金: {total_capital:,.0f} 元 | 持仓上限: {top_n} 只 | 成本: 0.3% 单边")
    print(f"{'=' * 70}")

    # 按权重排序
    portfolio = []
    for j, code in enumerate(common_codes):
        w = weights[j]
        if w >= 0.005:
            name = name_map.get(str(code).zfill(6), "未知")
            price = prices.get(code, 0)
            amount = total_capital * w
            shares = int(amount / price / 100) * 100
            portfolio.append({
                "code": code, "name": name, "weight": w,
                "price": price, "amount": amount, "shares": shares,
            })

    portfolio.sort(key=lambda x: x["weight"], reverse=True)

    print(f"\n  {'代码':8s} {'名称':10s} {'权重':>7s} {'最新价':>8s} {'建议金额':>10s} {'建议股数':>8s}")
    print(f"  {'-' * 60}")
    for p in portfolio:
        print(f"  {p['code']:8s} {p['name']:10s} {p['weight']*100:>6.1f}% "
              f"{p['price']:>8.2f} {p['amount']:>9.0f}元 {p['shares']:>6d}股")

    # 汇总
    total_w = sum(p["weight"] for p in portfolio)
    total_a = sum(p["amount"] for p in portfolio)
    print(f"  {'-' * 60}")
    print(f"  {'合计':18s} {total_w*100:>6.1f}% {'':>8s} {total_a:>9.0f}元")

    # 仓位分布
    heavy = [p for p in portfolio if p["weight"] >= 0.08]
    mid = [p for p in portfolio if 0.03 <= p["weight"] < 0.08]
    light = [p for p in portfolio if p["weight"] < 0.03]

    print(f"\n  仓位分布:")
    print(f"  重仓 (>=8%):  {len(heavy)} 只, 合计 {sum(p['weight'] for p in heavy)*100:.1f}%")
    print(f"  中仓 (3-8%):  {len(mid)} 只, 合计 {sum(p['weight'] for p in mid)*100:.1f}%")
    print(f"  轻仓 (<3%):   {len(light)} 只, 合计 {sum(p['weight'] for p in light)*100:.1f}%")

    # 行业分布
    print(f"\n  策略特征:")
    selected_names = [f.replace("factor_", "").replace("_raw", "") for f in selected_factors]
    print(f"  入选因子: {', '.join(selected_names[:8])}")
    print(f"  调仓建议: 每月末重新评估，按权重再平衡")
    print(f"  止损建议: 单只股票亏损超15%止损，组合回撤超10%减仓至50%")

    # 保存推荐
    rec_df = pd.DataFrame(portfolio)
    rec_df.to_csv(ROOT / "data" / "today_recommend.csv", index=False, encoding="utf-8-sig")
    print(f"\n推荐已保存: data/today_recommend.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
