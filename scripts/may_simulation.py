"""
2026年5月 模拟实盘 — 10万元 — 严格禁止未来信息

用法:
    python scripts/may_simulation.py
"""

import sys
import io

# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factors import compute_daily_factors, _align_financials, load_all


def estimate_covariance(ret_matrix, shrinkage=0.2):
    n = ret_matrix.shape[1]
    sample_cov = np.cov(ret_matrix, rowvar=False)
    diag_vals = np.diag(sample_cov)
    sqrt_d = np.sqrt(np.maximum(diag_vals, 0))
    target = np.outer(sqrt_d, sqrt_d)
    np.fill_diagonal(target, diag_vals)
    return (1 - shrinkage) * sample_cov + shrinkage * target


def max_sharpe_weights(mu, cov, max_w=0.10, min_w=0.01):
    n = len(mu)
    def neg_sharpe(w):
        pr = np.dot(w, mu)
        pv = np.sqrt(max(np.dot(w, np.dot(cov, w)), 1e-10))
        return -pr / pv
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w) for _ in range(n)]
    x0 = np.ones(n) / n
    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 500, "ftol": 1e-8})
    if result.success:
        w = np.maximum(result.x, 0)
        s = w.sum()
        return w / s if s > 0 else x0
    return x0


def ic_analysis(train_df):
    factor_cols = [c for c in train_df.columns if c.startswith("factor_")]
    results = []
    for col in factor_cols:
        ic_list = []
        for date, grp in train_df.groupby("date"):
            valid = grp[[col, "forward_ret_1m"]].dropna()
            if len(valid) >= 10:
                ic, _ = stats.spearmanr(valid[col], valid["forward_ret_1m"])
                ic_list.append(ic)
        if len(ic_list) >= 6:
            ic_arr = np.array(ic_list)
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr, ddof=1)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
            results.append({"factor": col, "ic_mean": ic_mean, "ic_ir": ic_ir, "t_stat": t_stat})
    return pd.DataFrame(results).sort_values("ic_ir", key=abs, ascending=False)


def build_factor_table():
    """构建含收盘价和 forward_ret_1m 的月度因子表"""
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
    month_end["forward_close"] = month_end.groupby("code")["close"].shift(-1)
    month_end["forward_ret_1m"] = month_end["forward_close"] / month_end["close"] - 1
    month_end = month_end[month_end["date"] >= pd.Timestamp("2020-01-01")]

    factor_cols = [c for c in month_end.columns if c.startswith("factor_")]
    out_cols = ["date", "code", "close"] + factor_cols + ["forward_ret_1m"]
    result = month_end[out_cols].dropna(subset=["forward_ret_1m"]).copy()
    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    for col in tqdm(factor_cols, desc="横截面标准化"):
        result[col] = result.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
    return result


def run_may_simulation(initial_capital=100000.0, top_n=30, transaction_cost=0.003):
    print("=" * 65)
    print(f"  2026年5月 模拟实盘 — 10万元")
    print(f"  初始本金: ¥{initial_capital:,.0f} | 持仓 top {top_n} | 单边成本: {transaction_cost*100:.1f}%")
    print("=" * 65)

    # ---- 1. 构建因子表 ----
    print("\n[1/5] 构建因子表（含 forward_ret_1m 未来收益标签）...")
    factor_table = build_factor_table()
    factor_cols_all = [c for c in factor_table.columns if c.startswith("factor_")]
    all_dates = sorted(factor_table["date"].unique())
    print(f"  因子表: {factor_table['date'].nunique()} 个月底, {factor_table['code'].nunique()} 只股票")
    print(f"  数据范围: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")

    # ---- 2. 确定决策日期 ----
    # 4月最后一个交易日 = 5月之前最大的日期
    april_dates = [d for d in all_dates if d < pd.Timestamp("2026-05-01")]
    if not april_dates:
        print("错误: 没有4月的数据!")
        return
    decision_date = april_dates[-1]  # 4月底
    # 5月最后一个交易日用于结算
    may_dates = [d for d in all_dates if d >= pd.Timestamp("2026-05-01") and d <= pd.Timestamp("2026-05-31")]
    settle_date = may_dates[-1] if may_dates else decision_date

    print(f"\n[2/5] 决策日期: {decision_date.strftime('%Y-%m-%d')} (4月末)")
    print(f"  结算日期: {settle_date.strftime('%Y-%m-%d')} (5月末)")
    print(f"  严格训练截止: 仅使用 < 2026-05-01 的数据")

    # ---- 3. 因子训练 (只用严格 < 2026-05-01 的数据) ----
    print("\n[3/5] 因子IC训练（仅用5月之前的数据）...")
    train_data = factor_table[factor_table["date"] < pd.Timestamp("2026-05-01")].copy()
    print(f"  训练数据: {train_data['date'].nunique()} 个月, {train_data['code'].nunique()} 只股票")

    ic_report = ic_analysis(train_data)
    print(f"\n  IC分析结果 (Top 15 因子):")
    print(f"  {'因子':30s} {'IC均值':>8s} {'IC_IR':>8s} {'t值':>8s}")
    print(f"  {'-' * 55}")
    for _, row in ic_report.head(15).iterrows():
        fname = row["factor"].replace("factor_", "")
        print(f"  {fname:30s} {row['ic_mean']:>+7.4f} {row['ic_ir']:>+7.3f} {row['t_stat']:>+7.2f}")

    # 选显著因子
    significant = ic_report[
        (abs(ic_report["ic_ir"]) > 0.08) | (abs(ic_report["t_stat"]) > 1.2)
    ]
    if len(significant) < 5:
        significant = ic_report.head(8)
    print(f"\n  显著因子: {len(significant)} 个")

    # 去相关
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

    print(f"  去相关后选中: {len(selected_factors)} 个因子")
    for f in selected_factors:
        row = ic_report[ic_report["factor"] == f]
        ic_ir_val = row["ic_ir"].values[0] if len(row) > 0 else 0
        sign = "+" if row["ic_mean"].values[0] >= 0 else "-" if len(row) > 0 else "?"
        print(f"    {f.replace('factor_', ''):35s} IC_IR={ic_ir_val:+.3f} 方向={sign}")

    # IC符号
    ic_signs = {}
    for _, row in ic_report.iterrows():
        ic_signs[row["factor"]] = 1 if row["ic_mean"] >= 0 else -1

    # ---- 4. 选股打分 ----
    print(f"\n[4/5] 4月末截面打分 & 选股...")
    current_month = factor_table[factor_table["date"] == decision_date].copy()
    print(f"  4月末截面: {len(current_month)} 只股票")

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

    # 流动性过滤: 只选有足够成交额的股票
    if "factor_amount" in current_month.columns:
        current_month = current_month[current_month["factor_amount"] > -3].copy()

    current_month = current_month.sort_values("composite_score", ascending=False)
    n_select = min(top_n, len(current_month))
    selected = current_month.head(n_select * 2)  # 先多选再优化

    # 用过去12个月的收益估算协方差
    past_dates = sorted(train_data["date"].unique()[-12:])
    past_returns = train_data[train_data["date"].isin(past_dates)]

    ret_data = {}
    prices = {}
    selected_codes_all = selected["code"].tolist()
    for code in selected_codes_all:
        stock_rets = past_returns[past_returns["code"] == code][["date", "forward_ret_1m"]]
        ret_data[code] = stock_rets.set_index("date")["forward_ret_1m"]
        price_row = selected[selected["code"] == code]
        if len(price_row) > 0:
            prices[code] = price_row["close"].iloc[0]

    ret_df = pd.DataFrame(ret_data).dropna(axis=1)
    common_codes = [c for c in selected_codes_all if c in ret_df.columns]

    if len(common_codes) >= 5:
        cov = estimate_covariance(ret_df[common_codes].values)
        mu = ret_df[common_codes].mean().values
        weights = max_sharpe_weights(mu, cov)
    else:
        common_codes = selected_codes_all[:min(len(selected_codes_all), top_n)]
        weights = np.ones(len(common_codes)) / len(common_codes)

    weights = weights / weights.sum()

    # 加载名称映射
    names = {}
    name_file = ROOT / "data" / "stock_names.csv"
    if name_file.exists():
        name_df = pd.read_csv(name_file)
        if "code" in name_df.columns and "name" in name_df.columns:
            names = dict(zip(name_df["code"].astype(str), name_df["name"]))

    # 加载行业映射
    industries = {}
    ind_file = ROOT / "data" / "industry_map.csv"
    if ind_file.exists():
        ind_df = pd.read_csv(ind_file)
        if "code" in ind_df.columns:
            ind_col = "industry" if "industry" in ind_df.columns else ind_df.columns[1]
            industries = dict(zip(ind_df["code"].astype(str), ind_df[ind_col]))

    # 构建最终持仓
    portfolio = []
    for j, code in enumerate(common_codes):
        w = weights[j]
        if w >= 0.005:
            name = names.get(str(code), "")
            ind = industries.get(str(code), "")
            price = prices.get(code, 0)
            amount = initial_capital * w
            shares = int(amount / price / 100) * 100  # 整手
            actual_amount = shares * price
            actual_weight = actual_amount / initial_capital
            portfolio.append({
                "code": code, "name": name, "industry": ind,
                "close": round(price, 2),
                "score": round(float(selected[selected["code"]==code]["composite_score"].values[0]), 4) if len(selected[selected["code"]==code]) > 0 else 0,
                "weight": round(w * 100, 2),
                "shares": shares,
                "amount": round(actual_amount, 2),
            })

    # 扣除交易成本
    cost = initial_capital * transaction_cost
    invested = sum(p["amount"] for p in portfolio)

    print(f"\n  建仓结果: {len(portfolio)} 只股票")
    print(f"  投入资金: ¥{invested:,.2f}")
    print(f"  交易成本: ¥{cost:,.2f}")
    print(f"  剩余现金: ¥{initial_capital - invested - cost:,.2f}")

    print(f"\n  {'代码':>8s}  {'名称':<8s}  {'行业':<10s}  {'买入价':>8s}  {'权重':>6s}  {'股数':>6s}  {'金额':>10s}")
    print(f"  {'-' * 68}")
    for p in sorted(portfolio, key=lambda x: x["weight"], reverse=True):
        print(f"  {p['code']:>8s}  {p['name']:<8s}  {p['industry']:<10s}  {p['close']:>8.2f}  {p['weight']:>5.1f}%  {p['shares']:>6d}  ¥{p['amount']:>9,.2f}")

    # ---- 5. 计算5月实际收益 ----
    print(f"\n[5/5] 计算5月实际收益...")
    # forward_ret_1m 在 decision_date (4月底) 的值 = 5月实际收益
    forward_ret_map = {}
    decision_data = factor_table[factor_table["date"] == decision_date]
    for _, row in decision_data.iterrows():
        forward_ret_map[str(row["code"])] = float(row["forward_ret_1m"])

    # 同时获取5月底的实际收盘价做验证
    settle_data = factor_table[factor_table["date"] == settle_date]
    settle_price_map = {}
    for _, row in settle_data.iterrows():
        settle_price_map[str(row["code"])] = float(row["close"])

    total_return = 0.0
    final_value = initial_capital - invested - cost  # 剩余现金

    print(f"\n  {'代码':>8s}  {'名称':<8s}  {'买入价':>8s}  {'卖出价':>8s}  {'收益%':>8s}  {'盈亏¥':>10s}  {'权重贡献%':>10s}")
    print(f"  {'-' * 78}")

    for p in portfolio:
        code = p["code"]
        ret = forward_ret_map.get(code, 0)
        buy_price = p["close"]
        sell_price = buy_price * (1 + ret)

        pnl = p["amount"] * ret
        final_value += p["amount"] + pnl
        total_return += (p["weight"] / 100) * ret

        print(f"  {code:>8s}  {p['name']:<8s}  {buy_price:>8.2f}  {sell_price:>8.2f}  {ret*100:>+7.2f}%  {pnl:>+9,.2f}  {(p['weight']/100)*ret*100:>+9.3f}%")

    total_profit = final_value - initial_capital
    total_pct = (final_value / initial_capital - 1) * 100
    # 不用复利，单月简单计算
    monthly_return = total_return * 100

    # 同期基准: 沪深300 5月表现 (用所有股票的等权平均近似)
    all_may_rets = []
    decision_data_valid = decision_data.dropna(subset=["forward_ret_1m"])
    if "factor_amount" in decision_data_valid.columns:
        decision_data_valid = decision_data_valid[decision_data_valid["factor_amount"] > -3]
    all_may_rets = decision_data_valid["forward_ret_1m"].dropna().tolist()
    benchmark_ret = np.mean(all_may_rets) * 100 if all_may_rets else 0

    print(f"\n{'=' * 65}")
    print(f"  2026年5月 模拟实盘结果")
    print(f"{'=' * 65}")
    print(f"  初始本金:       ¥{initial_capital:>12,.2f}")
    print(f"  最终资产:       ¥{final_value:>12,.2f}")
    print(f"  总收益:         ¥{total_profit:>+12,.2f}  ({total_pct:+.2f}%)")
    print(f"  其中 -> 持仓收益: {monthly_return:+.2f}%")
    print(f"         交易成本:  -{transaction_cost*100:.1f}%")
    print(f"  ─────────────────────────────────")
    print(f"  全A等权基准:     {benchmark_ret:+.2f}%")
    print(f"  超额收益(alpha): {total_pct - benchmark_ret:+.2f}%")
    print(f"{'=' * 65}")

    # 输出持仓明细以便追踪
    print(f"\n  5月选股因子贡献分析:")
    for f in selected_factors:
        f_short = f.replace("factor_", "")
        ic_row = ic_report[ic_report["factor"] == f]
        if len(ic_row) > 0:
            print(f"    {f_short:35s} IC={ic_row['ic_mean'].values[0]:+.4f}  "
                  f"IC_IR={ic_row['ic_ir'].values[0]:+.3f}  "
                  f"方向={'做多' if ic_signs.get(f,1)>0 else '做空'}")

    return portfolio, final_value, total_profit


if __name__ == "__main__":
    run_may_simulation(initial_capital=100000.0, top_n=30)
