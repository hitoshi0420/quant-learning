"""
今日开盘推荐 (使用最新日线数据)
用法: python today_recommend_v2.py
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
                    if not np.isnan(ic):
                        ic_list.append(ic)
                except Exception as e:
                    print(f"[today_recommend_v2] spearmanr {col} 失败: {e}")
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


def build_factor_table_full(start_date="2020-01-01"):
    """
    构建因子表 — 包含月末数据(用于训练IC) + 最新日线(用于当前选股)
    返回: (monthly_factor_table, latest_daily_factors, latest_date)
    """
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

    # ---- 月末因子表 (用于IC训练) ----
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_dates = df.groupby("year_month")["date"].max().reset_index(drop=True)
    monthly_dates = sorted(monthly_dates.tolist())

    month_end = df[df["date"].isin(monthly_dates)].copy()
    month_end = month_end.sort_values(["code", "date"])
    month_end["forward_close"] = month_end.groupby("code")["close"].shift(-1)
    month_end["forward_ret_1m"] = month_end["forward_close"] / month_end["close"] - 1
    month_end = month_end[month_end["date"] >= pd.Timestamp(start_date)]

    factor_cols_all = [c for c in month_end.columns if c.startswith("factor_")]
    out_cols = ["date", "code", "close"] + factor_cols_all + ["forward_ret_1m"]
    monthly_table = month_end[out_cols].dropna(subset=["forward_ret_1m"]).copy()
    monthly_table = monthly_table.sort_values(["date", "code"]).reset_index(drop=True)

    for col in tqdm(factor_cols_all, desc="标准化月末因子"):
        monthly_table[col] = monthly_table.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    # ---- 最新日线因子 (用于当前选股, 不需要 forward_ret_1m) ----
    latest_date = df["date"].max()
    latest_daily = df[df["date"] == latest_date].copy()
    # 用月末的均值和标准差做标准化 (避免未来信息)
    latest_daily = latest_daily.sort_values(["code"]).reset_index(drop=True)

    return monthly_table, latest_daily, factor_cols_all, latest_date


def main():
    print("=" * 70)
    print("  今日开盘推荐 — 多因子量化策略")
    print("=" * 70)

    # 1. 构建数据
    print("\n[1/4] 构建因子表 (月末训练 + 最新日线选股)...")
    monthly_table, latest_daily, factor_cols_all, latest_date = build_factor_table_full()

    print(f"  月末因子表: {monthly_table['date'].nunique()} 个月, {monthly_table['code'].nunique()} 只")
    print(f"  最新日线日期: {latest_date.strftime('%Y-%m-%d')} (共 {len(latest_daily)} 只股票)")
    print(f"  运行日期: {datetime.now().strftime('%Y-%m-%d')}")

    # 2. IC分析 (只用月末历史数据, latest_date之前)
    print(f"\n[2/4] IC分析 (训练数据截止 {latest_date.strftime('%Y-%m-%d')} 之前)...")
    train_data = monthly_table[monthly_table["date"] < latest_date].copy()
    ic_report = ic_analysis_train(train_data)

    # 只看有实际IC的因子
    ic_valid = ic_report[ic_report["ic_ir"].abs() > 0.001].copy()
    print(f"\n  {'因子':25s} {'IC均值':>8s} {'IC_IR':>8s} {'T值':>8s} {'IC次数':>8s}")
    print(f"  {'-' * 60}")
    for _, row in ic_valid.head(15).iterrows():
        name = row["factor"].replace("factor_", "").replace("_raw", "")
        print(f"  {name:25s} {row['ic_mean']:>+7.4f} {row['ic_ir']:>+7.3f} {row['t_stat']:>+7.2f}")

    # 3. 筛选显著因子
    significant = ic_valid[
        (abs(ic_valid["ic_ir"]) > 0.08) |
        (abs(ic_valid["t_stat"]) > 1.2)
    ]
    if len(significant) < 5:
        significant = ic_valid.head(8)

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
        selected_factors = [ic_valid.iloc[j]["factor"] for j in range(min(5, len(ic_valid)))]
        selected_factors = [f for f in selected_factors if f in factor_cols_all]

    ic_signs = {}
    for _, row in ic_valid.iterrows():
        ic_signs[row["factor"]] = 1 if row["ic_mean"] >= 0 else -1

    print(f"\n[3/4] 入选因子 ({len(selected_factors)} 个):")
    for f in selected_factors:
        name = f.replace("factor_", "").replace("_raw", "")
        row = ic_valid[ic_valid["factor"] == f]
        if len(row) > 0:
            print(f"  {name:30s} IC_IR={row['ic_ir'].values[0]:+.3f}  IC_mean={row['ic_mean'].values[0]:+.4f}")

    # 4. 最新日线选股打分
    print(f"\n[4/4] 最新日线 ({latest_date.strftime('%Y-%m-%d')}) 选股 + Max Sharpe 优化...")

    # 用月末因子表的均值和标准差来标准化最新日线的因子值
    current = latest_daily.copy()
    for col in selected_factors:
        if col in monthly_table.columns and col in current.columns:
            hist_mean = monthly_table[col].mean()
            hist_std = monthly_table[col].std()
            if hist_std > 0:
                current[col] = (current[col] - hist_mean) / hist_std
            else:
                current[col] = 0

    # 打分
    score = np.zeros(len(current))
    count = 0
    for col in selected_factors:
        if col in current.columns:
            vals = current[col].fillna(0).values
            if ic_signs.get(col, 1) < 0:
                vals = -vals
            score += vals
            count += 1
    current["composite_score"] = score / max(count, 1)
    current = current.sort_values("composite_score", ascending=False)

    top_n = 30
    selected = current.head(top_n)
    selected_codes = selected["code"].tolist()

    # 历史收益协方差 (用最近12个月月末数据)
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

    if len(common_codes) >= 5:
        cov = estimate_covariance(ret_df[common_codes].values)
        mu = ret_df[common_codes].mean().values
        weights = max_sharpe_weights(mu, cov)
    else:
        common_codes = selected_codes[:min(len(selected_codes), top_n)]
        weights = np.ones(len(common_codes)) / len(common_codes)

    weights = weights / weights.sum()

    # ---- 输出 ----
    total_capital = 10000.0

    print(f"\n{'=' * 70}")
    print(f"  今日推荐组合 ({datetime.now().strftime('%Y-%m-%d')} 开盘)")
    print(f"  数据截止: {latest_date.strftime('%Y-%m-%d')} | 基准本金: {total_capital:,.0f} 元")
    print(f"{'=' * 70}")

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
    print(f"  {'-' * 65}")
    for p in portfolio:
        print(f"  {p['code']:8s} {p['name']:10s} {p['weight']*100:>6.1f}% "
              f"{p['price']:>8.2f} {p['amount']:>9.0f}元 {p['shares']:>6d}股")

    total_w = sum(p["weight"] for p in portfolio)
    total_a = sum(p["amount"] for p in portfolio)
    print(f"  {'-' * 65}")
    print(f"  {'合计':18s} {total_w*100:>6.1f}% {'':>8s} {total_a:>9.0f}元")

    # 仓位分布
    heavy = [p for p in portfolio if p["weight"] >= 0.08]
    mid = [p for p in portfolio if 0.03 <= p["weight"] < 0.08]
    light = [p for p in portfolio if p["weight"] < 0.03]
    print(f"\n  仓位: 重仓{len(heavy)}只({sum(p['weight'] for p in heavy)*100:.0f}%) | "
          f"中仓{len(mid)}只({sum(p['weight'] for p in mid)*100:.0f}%) | "
          f"轻仓{len(light)}只({sum(p['weight'] for p in light)*100:.0f}%)")

    # 行业分布
    print(f"\n  入选因子: {', '.join(f.replace('factor_','').replace('_raw','') for f in selected_factors[:8])}")
    print(f"  策略类型: 月频调仓 | 单边成本0.3% | Max Sharpe权重优化")
    print(f"  风控: 个股权重上限10%, 下限1%, 长期满仓不做择时")

    rec_df = pd.DataFrame(portfolio)
    rec_df.to_csv(ROOT / "data" / "today_recommend.csv", index=False, encoding="utf-8-sig")
    print(f"\n推荐已保存: data/today_recommend.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
