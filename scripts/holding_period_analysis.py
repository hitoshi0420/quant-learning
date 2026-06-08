"""
持有期优化 — 测试 1~12 个月持有期下策略的累积收益
严格: 持有N个月 → 每N个月调仓一次, forward_ret 不重叠
优化版: 只加载一次数据, 避免重复计算因子

用法: python scripts/holding_period_analysis.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import compute_daily_factors, _align_financials, load_all


def estimate_covariance(ret_matrix, shrinkage=0.2):
    n = ret_matrix.shape[1]
    if n <= 1:
        return np.eye(n)
    sample_cov = np.cov(ret_matrix, rowvar=False)
    diag_vals = np.diag(sample_cov)
    sqrt_d = np.sqrt(np.maximum(diag_vals, 0))
    target = np.outer(sqrt_d, sqrt_d)
    np.fill_diagonal(target, diag_vals)
    return (1 - shrinkage) * sample_cov + shrinkage * target


def max_sharpe_weights(mu, cov, max_w=0.10, min_w=0.01):
    n = len(mu)
    if n <= 1:
        return np.ones(n) / n
    def neg_sharpe(w):
        pr = np.dot(w, mu)
        pv = np.sqrt(max(np.dot(w, np.dot(cov, w)), 1e-10))
        return -pr / pv
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w) for _ in range(n)]
    x0 = np.ones(n) / n
    try:
        result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 500, "ftol": 1e-8})
        if result.success:
            w = np.maximum(result.x, 0)
            s = w.sum()
            return w / s if s > 0 else x0
    except Exception:
        pass
    return x0


def ic_analysis(train_df):
    factor_cols = [c for c in train_df.columns if c.startswith("factor_")]
    results = []
    for col in factor_cols:
        ic_list = []
        for date, grp in train_df.groupby("date"):
            valid = grp[[col, "forward_ret"]].dropna()
            if len(valid) >= 10:
                try:
                    ic, _ = stats.spearmanr(valid[col], valid["forward_ret"])
                    if not np.isnan(ic):
                        ic_list.append(ic)
                except Exception:
                    pass
        if len(ic_list) >= 6:
            ic_arr = np.array(ic_list)
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr, ddof=1)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
            results.append({"factor": col, "ic_mean": ic_mean, "ic_ir": ic_ir, "t_stat": t_stat})
    return pd.DataFrame(results).sort_values("ic_ir", key=abs, ascending=False)


def build_base_table():
    """构建不含 forward_ret 的基础月度因子表 (只执行一次)"""
    print("  加载数据 & 计算因子 (一次性)...")
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
        on=["date", "code"], how="left")
    for col in available_fin.values():
        df[col] = df.groupby("code")[col].transform(lambda x: x.ffill())

    df["year_month"] = df["date"].dt.to_period("M")
    monthly_dates = df.groupby("year_month")["date"].max().reset_index(drop=True)
    monthly_dates = sorted(monthly_dates.tolist())

    month_end = df[df["date"].isin(monthly_dates)].copy()
    month_end = month_end.sort_values(["code", "date"])

    factor_cols = [c for c in month_end.columns if c.startswith("factor_")]
    out_cols = ["date", "code", "close"] + factor_cols
    result = month_end[out_cols].copy()
    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    for col in factor_cols:
        result[col] = result.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    print(f"  基础表: {result['date'].nunique()} 个月底, {result['code'].nunique()} 只股票")
    return result


def add_forward_ret(base_table, forward_months):
    """在基础表上添加 N 月 forward_ret, 过滤 >= 2020-01-01"""
    tbl = base_table.copy()
    tbl["forward_close"] = tbl.groupby("code")["close"].shift(-forward_months)
    tbl["forward_ret"] = tbl["forward_close"] / tbl["close"] - 1
    tbl = tbl[tbl["date"] >= pd.Timestamp("2020-01-01")]
    tbl = tbl.dropna(subset=["forward_ret"])
    tbl = tbl.sort_values(["date", "code"]).reset_index(drop=True)
    return tbl


def run_walkforward(factor_table, rebalance_months=1, start_date="2021-01-01", top_n=30, cost=0.003):
    """Walk-forward 回测, 每 rebalance_months 调仓一次"""
    all_dates = sorted(factor_table["date"].unique())
    factor_cols_all = [c for c in factor_table.columns if c.startswith("factor_")]

    sim_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
    rebalance_dates = [sim_dates[i] for i in range(0, len(sim_dates), rebalance_months)]

    if len(rebalance_dates) < 3:
        return 1.0, [], []

    capital = 1.0
    nav_history = [(rebalance_dates[0], 1.0)]
    prev_portfolio = {}

    for i, current_date in enumerate(rebalance_dates):
        if i > 0:
            prev_date = rebalance_dates[i - 1]
            prev_data = factor_table[factor_table["date"] == prev_date]
            ret_map = {}
            for _, row in prev_data.iterrows():
                ret_map[str(row["code"])] = float(row["forward_ret"])

            portfolio_return = sum(w * ret_map.get(code, 0) for code, w in prev_portfolio.items())
            capital *= (1 + portfolio_return)
            nav_history.append((current_date, capital))

        train_data = factor_table[factor_table["date"] < current_date].copy()
        if train_data["date"].nunique() < 12:
            prev_portfolio = {}
            continue

        ic_report = ic_analysis(train_data)
        if len(ic_report) < 3:
            prev_portfolio = {}
            continue

        significant = ic_report[
            (abs(ic_report["ic_ir"]) > 0.08) | (abs(ic_report["t_stat"]) > 1.2)
        ]
        if len(significant) < 5:
            significant = ic_report.head(8)

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

        current_month = factor_table[factor_table["date"] == current_date].copy()
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
        n_select = min(top_n * 2, len(current_month))
        selected = current_month.head(n_select)
        selected_codes = selected["code"].tolist()

        past_dates = sorted(train_data["date"].unique()[-12:])
        past_returns = train_data[train_data["date"].isin(past_dates)]

        ret_data = {}
        for code in selected_codes:
            stock_rets = past_returns[past_returns["code"] == code][["date", "forward_ret"]]
            ret_data[code] = stock_rets.set_index("date")["forward_ret"]

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

        portfolio = {}
        for j, code in enumerate(common_codes):
            w = weights[j]
            if w >= 0.005:
                portfolio[code] = float(w)
        total_w = sum(portfolio.values())
        if total_w > 0:
            for code in list(portfolio.keys()):
                portfolio[code] /= total_w

        if i > 0 and prev_portfolio:
            turnover = sum(abs(portfolio.get(c, 0) - prev_portfolio.get(c, 0))
                          for c in set(list(portfolio) + list(prev_portfolio))) / 2
            capital *= (1 - cost * 2 * turnover)

        prev_portfolio = dict(portfolio)

    return capital, nav_history


def main():
    horizons = [1, 2, 3, 4, 5, 6, 9, 12]
    results = {}

    print("=" * 70)
    print("  持有期优化分析 (修正版 — 持有N月=每N月调仓)")
    print("  回测区间: 2021-01 ~ 2026-04 | 初始资本: 1.0 | top 30")
    print("=" * 70)

    # 只加载一次数据
    base_table = build_base_table()

    for forward_m in horizons:
        print(f"\n{'─' * 70}")
        print(f"  持有期 = {forward_m} 个月 (每 {forward_m} 月调仓)")
        print(f"{'─' * 70}")
        print(f"  构建 {forward_m} 月 forward_ret...")

        factor_table = add_forward_ret(base_table, forward_months=forward_m)
        n_months = factor_table["date"].nunique()
        n_stocks = factor_table["code"].nunique()
        print(f"  因子表: {n_months} 个月底, {n_stocks} 只股票")

        final_nav, nav_history = run_walkforward(factor_table, rebalance_months=forward_m)
        n_trades = len(nav_history) - 1
        total_return = (final_nav - 1) * 100

        if n_trades > 0 and len(nav_history) >= 2:
            years = (nav_history[-1][0] - nav_history[0][0]).days / 365.25
            ann_return = (final_nav ** (1 / max(years, 0.5)) - 1) * 100
            navs = [v for _, v in nav_history]
            period_rets = [navs[i] / navs[i-1] - 1 for i in range(1, len(navs))]
            if period_rets:
                vol = np.std(period_rets) * np.sqrt(12 / forward_m) * 100
                mean_period = np.mean(period_rets)
                sharpe = (mean_period * (12 / forward_m) * 100) / vol if vol > 0 else 0
                max_dd = 0
                peak = navs[0]
                for v in navs:
                    if v > peak:
                        peak = v
                    dd = (v - peak) / peak
                    if dd < max_dd:
                        max_dd = dd
            else:
                vol = sharpe = max_dd = 0
        else:
            ann_return = vol = sharpe = max_dd = 0

        results[forward_m] = {
            "final_nav": round(final_nav, 4),
            "total_return": round(total_return, 2),
            "ann_return": round(ann_return, 2),
            "volatility": round(vol, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "n_trades": n_trades,
        }

        print(f"  最终净值: {final_nav:.4f} | 总收益: {total_return:+.1f}% | "
              f"年化: {ann_return:+.1f}% | 夏普: {sharpe:.2f} | 最大回撤: {max_dd*100:.1f}% | "
              f"调仓: {n_trades}次")

    # ---- 汇总 ----
    print(f"\n{'=' * 70}")
    print(f"  持有期收益汇总 (修正版)")
    print(f"{'=' * 70}")
    print(f"  {'持有期':<8s} {'总收益':>8s} {'年化收益':>8s} {'波动率':>8s} {'夏普':>8s} {'最大回撤':>8s} {'调仓次数':>8s}")
    print(f"  {'─' * 65}")

    best_return = max(results, key=lambda h: results[h]["total_return"])
    best_sharpe = max(results, key=lambda h: results[h]["sharpe"])
    best_ann = max(results, key=lambda h: results[h]["ann_return"])

    for h in horizons:
        r = results[h]
        markers = []
        if h == best_return: markers.append("收益")
        if h == best_sharpe: markers.append("夏普")
        marker = " <- " + "/".join(markers) + "最高" if markers else ""
        print(f"  {h}个月     {r['total_return']:>+7.1f}% {r['ann_return']:>+7.1f}% "
              f"{r['volatility']:>7.1f}% {r['sharpe']:>7.2f} {r['max_drawdown']:>7.1f}% {r['n_trades']:>7d}{marker}")

    print(f"\n  结论:")
    print(f"    总收益最高: 持有 {best_return} 个月 (+{results[best_return]['total_return']:.1f}%)")
    print(f"    年化收益最高: 持有 {best_ann} 个月 (+{results[best_ann]['ann_return']:.1f}%)")
    print(f"    夏普最高: 持有 {best_sharpe} 个月 ({results[best_sharpe]['sharpe']:.2f})")


if __name__ == "__main__":
    main()
