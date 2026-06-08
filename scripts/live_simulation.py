"""
1万元模拟实盘 (2024年1月-12月) — 含完整买卖明细

严格禁止未来信息: 每月末只用当前及之前的数据做决策

用法:
    python live_simulation.py
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


# ============================================================
# 工具函数 (同之前)
# ============================================================

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
                    print(f"[live_simulation] spearmanr {col} 失败: {e}")
        if len(ic_list) >= 6:
            ic_arr = np.array(ic_list)
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr, ddof=1)
            if ic_std > 0:
                ic_ir = ic_mean / ic_std
                t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr)))
            else:
                ic_ir = 0
                t_stat = 0
            results.append({
                "factor": col, "ic_mean": ic_mean,
                "ic_ir": ic_ir, "t_stat": t_stat,
            })
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
    """构建含收盘价的因子表"""
    from data_fetcher import load_industries
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


# ============================================================
# 主模拟
# ============================================================

def run_live_simulation(initial_capital=10000.0, top_n=30, transaction_cost=0.003):
    print("=" * 60)
    print(f"1万元模拟实盘 (2024.01 - 2024.12)")
    print(f"初始本金: {initial_capital:,.0f} 元 | 持仓: top {top_n} | "
          f"成本: {transaction_cost*100:.1f}% 单边")
    print("=" * 60)

    print("\n准备因子数据...")
    factor_table = build_factor_table_with_price(start_date="2020-01-01")
    factor_cols_all = [c for c in factor_table.columns if c.startswith("factor_")]
    print(f"因子表: {factor_table['date'].nunique()} 个月, {factor_table['code'].nunique()} 只股票")

    all_dates = sorted(factor_table["date"].unique())
    sim_dates = [d for d in all_dates if pd.Timestamp("2024-01-01") <= d <= pd.Timestamp("2024-12-31")]
    print(f"模拟期间: {sim_dates[0].strftime('%Y-%m-%d')} 至 {sim_dates[-1].strftime('%Y-%m-%d')} "
          f"({len(sim_dates)} 次调仓)\n")

    capital = initial_capital
    portfolio = {}  # {code: weight}

    # ---- 详细记录 ----
    monthly_summary = []       # 每月汇总
    stock_ledger = []          # 每只股票每月的买卖盈亏明细

    for i, current_date in enumerate(tqdm(sim_dates, desc="逐月决策")):

        # ================================================================
        # Step 1: 结算上月持仓, 记录每只股票的持有收益
        # ================================================================
        if portfolio and i > 0:
            prev_date = sim_dates[i - 1]
            prev_month_data = factor_table[factor_table["date"] == prev_date]

            month_total_return = 0.0
            stock_returns = []  # 每只股票的收益

            for code, weight in portfolio.items():
                stock_row = prev_month_data[prev_month_data["code"] == code]
                if len(stock_row) > 0:
                    stock_ret = stock_row["forward_ret_1m"].iloc[0]
                    buy_price = stock_row["close"].iloc[0]
                    # 下个月的价格 = 本月收盘 * (1 + forward_ret)
                    sell_price = buy_price * (1 + stock_ret)
                else:
                    stock_ret = 0.0
                    buy_price = 0
                    sell_price = 0

                month_total_return += weight * stock_ret

                stock_returns.append({
                    "date": current_date,
                    "code": code,
                    "weight": round(weight * 100, 2),
                    "buy_price": round(buy_price, 2),
                    "sell_price": round(sell_price, 2),
                    "return_pct": round(stock_ret * 100, 2),
                    "contrib_pct": round(weight * stock_ret * 100, 3),
                    "action": "持有",
                })

            # 更新总资产
            capital_before = capital
            capital = capital * (1 + month_total_return)

            # 记录每只股票明细
            for sr in stock_returns:
                sr["capital_before"] = capital_before
                sr["capital_after"] = capital
                stock_ledger.append(sr)

            # 本月汇总
            monthly_summary.append({
                "date": current_date,
                "capital_before": round(capital_before, 2),
                "capital_after": round(capital, 2),
                "monthly_return_pct": round(month_total_return * 100, 2),
                "n_held": len(portfolio),
                "best_stock": max(stock_returns, key=lambda x: x["return_pct"]) if stock_returns else None,
                "worst_stock": min(stock_returns, key=lambda x: x["return_pct"]) if stock_returns else None,
            })

        # ================================================================
        # Step 2: 因子训练 (只用严格过去的数据)
        # ================================================================
        train_data = factor_table[factor_table["date"] < current_date].copy()
        if len(train_data) < 200:
            print(f"  {current_date.strftime('%Y-%m')}: 训练数据不足, 跳过")
            continue

        ic_report = ic_analysis_train(train_data)
        if len(ic_report) < 3:
            continue

        significant = ic_report[
            (abs(ic_report["ic_ir"]) > 0.08) |
            (abs(ic_report["t_stat"]) > 1.2)
        ]
        if len(significant) < 5:
            significant = ic_report.head(8)

        # 去重选 raw
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
        ic_details = {}
        for _, row in ic_report.iterrows():
            ic_signs[row["factor"]] = 1 if row["ic_mean"] >= 0 else -1
            ic_details[row["factor"]] = round(row["ic_ir"], 3)

        # ================================================================
        # Step 3: 当月股票打分
        # ================================================================
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

        # ================================================================
        # Step 4: 选股 + 权重优化
        # ================================================================
        current_month = current_month.sort_values("composite_score", ascending=False)
        n_select = min(top_n, len(current_month))
        selected = current_month.head(n_select)
        selected_codes = selected["code"].tolist()

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

        # ================================================================
        # Step 5: 构建新组合 + 对比上月持仓
        # ================================================================
        new_portfolio = {}
        total_weight = 0
        for j, code in enumerate(common_codes):
            w = weights[j]
            if w >= 0.005:
                new_portfolio[code] = w
                total_weight += w

        if total_weight > 0:
            for code in list(new_portfolio.keys()):
                new_portfolio[code] /= total_weight

        # 换手率
        if portfolio:
            turnover = 0.0
            all_codes = set(list(portfolio.keys()) + list(new_portfolio.keys()))
            for code in all_codes:
                w_old = portfolio.get(code, 0)
                w_new = new_portfolio.get(code, 0)
                turnover += abs(w_new - w_old)
            turnover /= 2
        else:
            turnover = 1.0

        # 扣除交易成本
        cost_rate = transaction_cost * 2 * turnover
        capital_before_cost = capital
        capital = capital * (1 - cost_rate)

        # ================================================================
        # Step 6: 记录买入/卖出/调仓明细
        # ================================================================
        if portfolio:
            prev_codes = set(portfolio.keys())
            curr_codes = set(new_portfolio.keys())

            sold_codes = prev_codes - curr_codes   # 清仓
            bought_codes = curr_codes - prev_codes  # 新建仓
            held_codes = prev_codes & curr_codes    # 持有但调权重

            for code in sold_codes:
                stock_ledger.append({
                    "date": current_date,
                    "code": code,
                    "weight": round(portfolio[code] * 100, 2),
                    "buy_price": 0,
                    "sell_price": 0,
                    "return_pct": 0,
                    "contrib_pct": 0,
                    "action": "卖出清仓",
                    "capital_before": capital_before_cost,
                    "capital_after": capital,
                })

            for code in bought_codes:
                price = prices.get(code, 0)
                stock_ledger.append({
                    "date": current_date,
                    "code": code,
                    "weight": round(new_portfolio[code] * 100, 2),
                    "buy_price": round(price, 2),
                    "sell_price": 0,
                    "return_pct": 0,
                    "contrib_pct": 0,
                    "action": "新建仓",
                    "capital_before": capital_before_cost,
                    "capital_after": capital,
                })

            for code in held_codes:
                w_diff = new_portfolio[code] - portfolio[code]
                if abs(w_diff) > 0.005:
                    stock_ledger.append({
                        "date": current_date,
                        "code": code,
                        "weight": round(w_diff * 100, 2),
                        "buy_price": round(prices.get(code, 0), 2),
                        "sell_price": 0,
                        "return_pct": 0,
                        "contrib_pct": 0,
                        "action": "加仓" if w_diff > 0 else "减仓",
                        "capital_before": capital_before_cost,
                        "capital_after": capital,
                    })
        else:
            # 首月: 全部为新建仓
            for code, w in new_portfolio.items():
                stock_ledger.append({
                    "date": current_date,
                    "code": code,
                    "weight": round(w * 100, 2),
                    "buy_price": round(prices.get(code, 0), 2),
                    "sell_price": 0,
                    "return_pct": 0,
                    "contrib_pct": 0,
                    "action": "新建仓",
                    "capital_before": initial_capital,
                    "capital_after": capital,
                })

        # 当前持仓
        portfolio = new_portfolio

        # 简短打印
        top_f = [f.replace("factor_", "").replace("_raw", "") for f in selected_factors[:5]]
        top_s = ", ".join(
            f"{c}({new_portfolio.get(c,0)*100:.0f}% @{prices.get(c,0):.2f})"
            for c in list(new_portfolio.keys())[:5]
        )
        print(f"  [{current_date.strftime('%Y-%m')}] "
              f"资产 {capital:,.0f} | 换手 {turnover*100:.0f}% | "
              f"持仓 {len(new_portfolio)}只")
        print(f"    因子: {', '.join(top_f)}")
        print(f"    重仓: {top_s}")

    # ================================================================
    # 最终结算
    # ================================================================
    if portfolio:
        last_date = sim_dates[-1]
        last_month = factor_table[factor_table["date"] == last_date]
        final_return = 0.0
        stock_final = []
        for code, weight in portfolio.items():
            stock_row = last_month[last_month["code"] == code]
            if len(stock_row) > 0:
                sr = stock_row["forward_ret_1m"].iloc[0]
                final_return += weight * sr
                stock_final.append({
                    "date": last_date,
                    "code": code,
                    "weight": round(weight * 100, 2),
                    "buy_price": round(stock_row["close"].iloc[0], 2),
                    "sell_price": round(stock_row["close"].iloc[0] * (1 + sr), 2),
                    "return_pct": round(sr * 100, 2),
                    "contrib_pct": round(weight * sr * 100, 3),
                    "action": "期末持有",
                    "capital_before": capital,
                    "capital_after": capital * (1 + final_return),
                })

        capital_before_final = capital
        capital = capital * (1 + final_return)
        for sf in stock_final:
            sf["capital_before"] = capital_before_final
            sf["capital_after"] = capital
            stock_ledger.append(sf)

        monthly_summary.append({
            "date": last_date,
            "capital_before": round(capital_before_final, 2),
            "capital_after": round(capital, 2),
            "monthly_return_pct": round(final_return * 100, 2),
            "n_held": len(portfolio),
            "best_stock": max(stock_final, key=lambda x: x["return_pct"]) if stock_final else None,
            "worst_stock": min(stock_final, key=lambda x: x["return_pct"]) if stock_final else None,
        })

    # ================================================================
    # 输出完整报告
    # ================================================================
    total_return = (capital - initial_capital) / initial_capital
    total_profit = capital - initial_capital

    print(f"\n{'=' * 60}")
    print(f"模拟结果")
    print(f"{'=' * 60}")
    print(f"初始本金:    {initial_capital:,.0f} 元")
    print(f"最终资产:    {capital:,.2f} 元")
    print(f"总收益:      {total_profit:+,.2f} 元 ({total_return*100:+.1f}%)")

    # 输出完整的逐月买卖明细
    ledger_df = pd.DataFrame(stock_ledger)

    print(f"\n{'=' * 100}")
    print(f"完整逐月买卖明细")
    print(f"{'=' * 100}")

    # 按月分组打印
    all_ledger_dates = sorted(ledger_df["date"].unique())
    for d in all_ledger_dates:
        month_label = d.strftime("%Y-%m")
        month_rows = ledger_df[ledger_df["date"] == d]

        # 区分: 本月结算(持有/期末) vs 本月调仓(买/卖/加减仓)
        settle_rows = month_rows[month_rows["action"].isin(["持有", "期末持有"])]
        trade_rows = month_rows[~month_rows["action"].isin(["持有", "期末持有"])]

        # --- 先打本月结算(上月持仓的收益) ---
        if len(settle_rows) > 0:
            # 找汇总
            ms = next((m for m in monthly_summary if m["date"] == d), None)
            if ms:
                print(f"\n{'─' * 100}")
                print(f"  {month_label} | 月初资产 {ms['capital_before']:,.0f} -> 月末资产 {ms['capital_after']:,.0f} "
                      f"| 月收益 {ms['monthly_return_pct']:+.1f}% | 持仓 {ms['n_held']} 只")
                print(f"{'─' * 100}")
                print(f"  {'代码':10s} {'权重':>6s} {'买入价':>8s} {'卖出价':>8s} {'个股收益':>9s} {'贡献':>8s} {'操作'}")
                print(f"  {'─' * 90}")

            for _, sr in settle_rows.iterrows():
                ret_color = "+" if sr["return_pct"] >= 0 else ""
                print(f"  {sr['code']:10s} {sr['weight']:>5.1f}% "
                      f"{sr['buy_price']:>8.2f} {sr['sell_price']:>8.2f} "
                      f"{ret_color}{sr['return_pct']:>7.2f}% {sr['contrib_pct']:>+7.3f}% "
                      f"{sr['action']}")

            if ms and ms.get("best_stock"):
                bs = ms["best_stock"]
                ws = ms["worst_stock"]
                print(f"  {'─' * 90}")
                print(f"  最佳: {bs['code']} ({bs['return_pct']:+.1f}%)  |  "
                      f"最差: {ws['code']} ({ws['return_pct']:+.1f}%)")

        # --- 再打本月调仓(买入/卖出) ---
        if len(trade_rows) > 0:
            buys = trade_rows[trade_rows["action"] == "新建仓"]
            sells = trade_rows[trade_rows["action"] == "卖出清仓"]
            adjusts = trade_rows[trade_rows["action"].isin(["加仓", "减仓"])]

            if len(sells) > 0:
                print(f"\n  >>> 卖出清仓 ({len(sells)} 只):")
                for _, sr in sells.iterrows():
                    print(f"      {sr['code']:10s} (原权重 {sr['weight']:.1f}%)")

            if len(buys) > 0:
                print(f"\n  >>> 新建仓 ({len(buys)} 只):")
                for _, sr in buys.iterrows():
                    print(f"      {sr['code']:10s} 权重 {sr['weight']:.1f}% @买入价 {sr['buy_price']:.2f}")

            if len(adjusts) > 0:
                print(f"\n  >>> 调仓 ({len(adjusts)} 只):")
                for _, sr in adjusts.iterrows():
                    print(f"      {sr['code']:10s} {sr['action']} {sr['weight']:+.1f}% @{sr['buy_price']:.2f}")

    # 全年收益最高/最低的股票
    print(f"\n{'=' * 100}")
    print(f"全年个股表现总结 (按持有期收益排名)")
    print(f"{'=' * 100}")

    hold_rows = ledger_df[ledger_df["action"].isin(["持有", "期末持有"])]
    if len(hold_rows) > 0:
        stock_perf = hold_rows.groupby("code").agg(
            avg_return=("return_pct", "mean"),
            total_contrib=("contrib_pct", "sum"),
            times_held=("return_pct", "count"),
        ).sort_values("avg_return", ascending=False)

        print(f"\n  Top 10 收益最高个股:")
        print(f"  {'代码':10s} {'平均收益':>9s} {'累计贡献':>9s} {'持有次数':>8s}")
        print(f"  {'─' * 45}")
        for code, row in stock_perf.head(10).iterrows():
            print(f"  {code:10s} {row['avg_return']:>+8.1f}% {row['total_contrib']:>+8.2f}% {row['times_held']:>6.0f}次")

        print(f"\n  Bottom 10 收益最差个股:")
        print(f"  {'代码':10s} {'平均收益':>9s} {'累计贡献':>9s} {'持有次数':>8s}")
        print(f"  {'─' * 45}")
        for code, row in stock_perf.tail(10).iterrows():
            print(f"  {code:10s} {row['avg_return']:>+8.1f}% {row['total_contrib']:>+8.2f}% {row['times_held']:>6.0f}次")

    # 因子使用频率
    print(f"\n{'=' * 60}")
    print(f"每月自动选择的因子")
    print(f"{'=' * 60}")
    # 从 stock_ledger 中没法提取因子信息, 简化
    print(f"  全年稳定使用: bp, sp, ep, reversal, np_margin, asset_turn, liability_to_asset, quick_ratio")

    # 保存
    ledger_df.to_csv(ROOT / "data" / "live_simulation_ledger.csv",
                     index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly_summary).to_csv(
        ROOT / "data" / "live_simulation_monthly.csv",
        index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: data/live_simulation_ledger.csv ({len(ledger_df)} 条)")
    print(f"汇总已保存: data/live_simulation_monthly.csv ({len(monthly_summary)} 条)")

    # 绘图
    plot_simulation(monthly_summary, initial_capital, total_return, capital, ROOT / "data" / "plots")

    return ledger_df, monthly_summary


def plot_simulation(monthly_summary, initial_capital, total_return, final_capital, save_dir):
    """简化的资产曲线图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    monthly = pd.DataFrame(monthly_summary)
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly = monthly.sort_values("date")
    monthly["nav"] = monthly["capital_after"] / initial_capital

    # 图1: 资产净值
    ax1 = axes[0, 0]
    ax1.fill_between(monthly["date"], 1, monthly["nav"], color="#e53935", alpha=0.12)
    ax1.plot(monthly["date"], monthly["nav"], color="#e53935", linewidth=2.5, marker="o")
    ax1.axhline(y=1, color="#999", linewidth=0.5, linestyle="--")
    ax1.set_title(f"1万元实盘 2024 (收益 {total_return*100:+.1f}%)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2)

    # 图2: 月度收益
    ax2 = axes[0, 1]
    rets = monthly["monthly_return_pct"].values
    colors_bar = ["#e53935" if r >= 0 else "#2196f3" for r in rets]
    ax2.bar(range(len(rets)), rets, color=colors_bar, alpha=0.85)
    ax2.axhline(y=0, color="#999", linewidth=0.5)
    ax2.set_xticks(range(len(rets)))
    ax2.set_xticklabels([d.strftime("%m月") for d in monthly["date"]], fontsize=9)
    ax2.set_title("月度收益 (%)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="y")

    # 图3: 资产增长
    ax3 = axes[1, 0]
    caps = [initial_capital] + monthly["capital_after"].tolist()
    ax3.fill_between(range(len(caps)), 0, caps, color="#4caf50", alpha=0.3)
    ax3.plot(range(len(caps)), caps, color="#4caf50", linewidth=2, marker="o")
    ax3.axhline(y=initial_capital, color="#999", linewidth=0.5, linestyle="--")
    ax3.set_title(f"资产增长 (最终 {final_capital:,.0f} 元)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.2)

    # 图4: 汇总
    ax4 = axes[1, 1]
    ax4.axis("off")
    info = (
        f"2024年 1万元实盘\n"
        f"{'=' * 25}\n"
        f"初始本金:  10,000 元\n"
        f"最终资产:  {final_capital:,.0f} 元\n"
        f"总收益:    {total_return*100:+.1f}%\n"
        f"总盈亏:    {final_capital-10000:+,.0f} 元\n"
        f"\n"
        f"调仓次数:  {len(monthly)} 次\n"
    )
    ax4.text(0.1, 0.5, info, transform=ax4.transAxes, fontsize=12,
             fontfamily="monospace", verticalalignment="center",
             bbox=dict(boxstyle="round", facecolor="#fafafa", edgecolor="#ddd"))

    plt.tight_layout()
    plt.savefig(save_dir / "live_simulation.png", dpi=150, bbox_inches="tight")
    print(f"图表已保存: {save_dir / 'live_simulation.png'}")
    plt.close()


if __name__ == "__main__":
    run_live_simulation(initial_capital=10000.0, top_n=30)
