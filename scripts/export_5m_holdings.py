"""
导出 5 个月持有期的具体持仓明细到 Excel
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
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
    print("加载数据 & 计算因子...")
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
    return result


def add_forward_ret(base_table, forward_months):
    tbl = base_table.copy()
    tbl["forward_close"] = tbl.groupby("code")["close"].shift(-forward_months)
    tbl["forward_ret"] = tbl["forward_close"] / tbl["close"] - 1
    tbl = tbl[tbl["date"] >= pd.Timestamp("2020-01-01")]
    tbl = tbl.dropna(subset=["forward_ret"])
    tbl = tbl.sort_values(["date", "code"]).reset_index(drop=True)
    return tbl


def main():
    REBALANCE_MONTHS = 5
    TOP_N = 30
    COST = 0.003

    base_table = build_base_table()
    factor_table = add_forward_ret(base_table, forward_months=REBALANCE_MONTHS)
    all_dates = sorted(factor_table["date"].unique())
    factor_cols_all = [c for c in factor_table.columns if c.startswith("factor_")]

    sim_dates = [d for d in all_dates if d >= pd.Timestamp("2021-01-01")]
    rebalance_dates = [sim_dates[i] for i in range(0, len(sim_dates), REBALANCE_MONTHS)]

    capital = 1.0
    prev_portfolio = {}

    # 加载股票名称映射
    print("加载股票名称...")
    stock_info = pd.read_csv(ROOT / "data" / "stock_names.csv", dtype=str, encoding="utf-8-sig")
    name_map = dict(zip(stock_info["code_clean"].astype(str), stock_info["name"]))

    all_trades = []  # 每行的交易记录

    for i, current_date in enumerate(rebalance_dates):
        # 结算上期
        if i > 0:
            prev_date = rebalance_dates[i - 1]
            prev_data = factor_table[factor_table["date"] == prev_date]
            ret_map = {}
            for _, row in prev_data.iterrows():
                ret_map[str(row["code"])] = float(row["forward_ret"])

            for code, w in prev_portfolio.items():
                actual_ret = ret_map.get(code, 0)
                stock_return_contrib = w * actual_ret
                all_trades.append({
                    "调仓日期": str(prev_date.date()),
                    "持有至": str(current_date.date()),
                    "股票代码": code,
                    "股票名称": name_map.get(code, "未知"),
                    "持仓权重": round(w * 100, 2),
                    "期间收益": round(actual_ret * 100, 2),
                    "对组合贡献": round(stock_return_contrib * 100, 2),
                    "期间净值": round(capital, 4),
                })

            portfolio_return = sum(w * ret_map.get(code, 0) for code, w in prev_portfolio.items())
            capital *= (1 + portfolio_return)

        # 训练选股
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
        n_select = min(TOP_N * 2, len(current_month))
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
            common_codes = selected_codes[:min(len(selected_codes), TOP_N)]
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
            capital *= (1 - COST * 2 * turnover)

        prev_portfolio = dict(portfolio)

    # 构建 DataFrame
    df_trades = pd.DataFrame(all_trades)
    if not df_trades.empty:
        df_trades = df_trades.sort_values(["调仓日期", "期间收益"], ascending=[True, False])

    # 写入 Excel
    output_path = ROOT / "data" / "持有5个月_持仓明细.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: 全部持仓明细
        df_trades.to_excel(writer, sheet_name="持仓明细", index=False)

        # Sheet 2: 每期汇总
        period_summary = df_trades.groupby("调仓日期").agg(
            持仓数=("股票代码", "count"),
            组合收益=("对组合贡献", "sum"),
            期末净值=("期间净值", "last"),
        ).reset_index()
        period_summary["组合收益"] = period_summary["组合收益"].round(2)
        period_summary.to_excel(writer, sheet_name="每期汇总", index=False)

        # Sheet 3: 股票出现频次
        freq = df_trades.groupby(["股票代码", "股票名称"]).agg(
            出现次数=("调仓日期", "nunique"),
            平均权重=("持仓权重", "mean"),
            平均收益=("期间收益", "mean"),
        ).reset_index()
        freq = freq.sort_values("出现次数", ascending=False)
        freq.to_excel(writer, sheet_name="股票频次统计", index=False)

    print(f"已导出: {output_path}")
    print(f"共 {len(df_trades)} 条持仓记录, {df_trades['调仓日期'].nunique()} 个持仓周期")


if __name__ == "__main__":
    main()
