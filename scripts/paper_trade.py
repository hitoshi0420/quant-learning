"""
实盘模拟：10万元，月度调仓，最少100股，禁止未来信息
用法: python scripts/paper_trade.py
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


# ============================================================
# 1. 加载数据
# ============================================================

def load_prices(dates, daily_dir):
    """从日线文件加载指定日期的收盘价 —— 只读需要的日期"""
    print("加载价格数据...")
    prices = {}
    date_set = set(pd.Timestamp(d) for d in dates)

    for f in tqdm(sorted(daily_dir.glob("*.parquet")), desc="价格"):
        try:
            df = pd.read_parquet(f, columns=["date", "close"])
            df["date"] = pd.to_datetime(df["date"])
            for _, row in df.iterrows():
                if row["date"] in date_set:
                    prices[(f.stem, row["date"])] = float(row["close"])
        except Exception:
            pass
    return prices


def main():
    print("=" * 70)
    print("  实盘模拟 — 10万元多因子月度调仓")
    print("=" * 70)

    # 加载因子表
    print("加载因子表...")
    ft = pd.read_parquet(ROOT / "data" / "factor_table_full.parquet")
    ft["date"] = pd.to_datetime(ft["date"])

    # 模拟期间：最近 12 次完整调仓周期
    all_dates = sorted(ft["date"].unique())
    # 取最近 13 个月（12 次调仓 + 最终估值日）
    sim_dates = all_dates[-13:]  # 2025-05-30 to 2026-04-30
    rebalance_dates = sim_dates[:-1]  # 12 个调仓日
    final_date = sim_dates[-1]  # 最终估值日

    print(f"模拟期间: {sim_dates[0].strftime('%Y-%m-%d')} ~ {final_date.strftime('%Y-%m-%d')}")
    print(f"调仓次数: {len(rebalance_dates)}")

    # 加载价格
    daily_dir = ROOT / "data" / "daily"
    prices = load_prices(sim_dates, daily_dir)
    print(f"价格条目: {len(prices)}")

    # ============================================================
    # 2. 因子定义
    # ============================================================

    quality_factors = [
        "factor_roe", "factor_np_margin",
        "factor_current_ratio", "factor_liability_to_asset",
        "factor_cfo_to_np",
    ]
    value_factors = ["factor_bp", "factor_sp"]
    tech_factors = ["factor_pe_momentum", "factor_reversal", "factor_vol_20d"]

    all_factor_cols = quality_factors + value_factors + tech_factors
    all_factor_cols = [c for c in all_factor_cols if c in ft.columns]
    print(f"使用因子 ({len(all_factor_cols)}个): {[c.replace('factor_','') for c in all_factor_cols]}")

    # ============================================================
    # 3. 模拟引擎
    # ============================================================

    INITIAL_CAPITAL = 100_000.0
    TOP_N = 10
    MIN_SHARES = 100
    COMMISSION_RATE = 0.0003   # 万三佣金
    STAMP_DUTY = 0.0005        # 卖出印花税 (仅卖出)
    SLIPPAGE = 0.001           # 滑点 0.1%

    # 加载股票名称
    try:
        names_df = pd.read_csv(ROOT / "data" / "stock_names.csv", dtype={"code_clean": str})
        name_map = dict(zip(names_df["code_clean"], names_df["name"]))
    except Exception:
        name_map = {}
    from data_fetcher import load_industries
    ind_df = load_industries()
    ind_map = dict(zip(ind_df["code"], ind_df["industry"]))

    cash = INITIAL_CAPITAL
    holdings = {}  # {code: {shares, cost_price, buy_date}}
    portfolio_values = []
    trades_log = []
    monthly_detail = []  # 每月详细记录

    print(f"\n{'=' * 80}")
    print(f"  开始模拟 — 初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print(f"  规则: 每月末调仓, top{TOP_N}只, 最少{MIN_SHARES}股, 佣金万三+印花税千0.5(卖)")
    print(f"{'=' * 80}")

    for i, rebal_date in enumerate(rebalance_dates):
        month_record = {"date": rebal_date, "sells": [], "buys": [], "holds": [],
                        "cash_before": round(cash, 2)}

        # ---- Step 1: 获取当日可用股票（有价格 + 有因子值 + Top800筛选） ----
        month_data = ft[ft["date"] == rebal_date].copy()
        if len(month_data) == 0:
            continue

        month_data = month_data.sort_values("factor_size", ascending=True).head(800)

        valid_codes = []
        for _, row in month_data.iterrows():
            code = row["code"]
            if (code, rebal_date) in prices and pd.notna(row["forward_ret_1m"]):
                valid_codes.append(code)

        month_data = month_data[month_data["code"].isin(valid_codes)].copy()
        if len(month_data) < TOP_N:
            continue

        # ---- Step 2: 计算因子得分 ----
        available_factors = [c for c in all_factor_cols if c in month_data.columns]
        for col in available_factors:
            month_data[col] = month_data[col].fillna(0)

        month_data["score"] = month_data[available_factors].mean(axis=1)
        month_data = month_data.sort_values("score", ascending=False)

        # 记录 top 10 因子得分
        selected = month_data.head(TOP_N)
        selected_codes = set(selected["code"].tolist())

        # ---- Step 3: 卖出 ----
        start_cash = cash
        sell_codes = [c for c in holdings if c not in selected_codes]
        sell_value = 0
        for code in sell_codes:
            h = holdings[code]
            sell_price = prices.get((code, rebal_date), h["cost_price"])
            actual_sell = sell_price * (1 - SLIPPAGE)
            proceeds = h["shares"] * actual_sell
            commission = max(proceeds * COMMISSION_RATE, 5)
            stamp = proceeds * STAMP_DUTY
            net_proceeds = proceeds - commission - stamp
            sell_value += net_proceeds

            pnl = net_proceeds - h["shares"] * h["cost_price"]
            month_record["sells"].append({
                "code": code, "name": name_map.get(code, ""),
                "shares": h["shares"], "price": round(sell_price, 2),
                "proceeds": round(net_proceeds, 2), "pnl": round(pnl, 2),
            })
            trades_log.append({
                "date": rebal_date, "code": code, "action": "卖出",
                "shares": h["shares"], "price": sell_price,
                "amount": proceeds, "commission": commission, "stamp": stamp,
                "net": net_proceeds,
            })
            del holdings[code]

        cash += sell_value
        month_record["sell_total"] = round(sell_value, 2)

        # ---- Step 4: 买入 ----
        target_per_stock = cash / TOP_N
        buy_value = 0

        # 先卖后买，all_target_codes 是 selected 的全部
        all_target_codes = [c for c in selected["code"].tolist()]

        for code in all_target_codes:
            buy_price = prices.get((code, rebal_date))
            if buy_price is None or buy_price <= 0:
                continue
            actual_buy = buy_price * (1 + SLIPPAGE)

            if code in holdings:
                # 记录继续持有
                h = holdings[code]
                current_price = prices.get((code, rebal_date), h["cost_price"])
                mkt_val = h["shares"] * current_price
                month_record["holds"].append({
                    "code": code, "name": name_map.get(code, ""),
                    "shares": h["shares"], "cost": round(h["cost_price"], 2),
                    "now": round(current_price, 2), "mkt_val": round(mkt_val, 2),
                })
                continue

            max_shares = int(target_per_stock / actual_buy / MIN_SHARES) * MIN_SHARES
            if max_shares < MIN_SHARES:
                continue

            cost = max_shares * actual_buy
            commission = max(cost * COMMISSION_RATE, 5)
            total_cost = cost + commission

            if total_cost > cash - buy_value:
                affordable = int((cash - buy_value) / (actual_buy * (1 + COMMISSION_RATE)) / MIN_SHARES) * MIN_SHARES
                if affordable < MIN_SHARES:
                    continue
                max_shares = affordable
                cost = max_shares * actual_buy
                commission = max(cost * COMMISSION_RATE, 5)
                total_cost = cost + commission

            holdings[code] = {"shares": max_shares, "cost_price": buy_price, "buy_date": rebal_date}
            buy_value += total_cost

            month_record["buys"].append({
                "code": code, "name": name_map.get(code, ""),
                "industry": ind_map.get(code, "未知"),
                "shares": max_shares, "price": round(buy_price, 2),
                "cost": round(total_cost, 2), "score": round(
                    selected[selected["code"]==code]["score"].values[0], 3,
                ),
            })
            trades_log.append({
                "date": rebal_date, "code": code, "action": "买入",
                "shares": max_shares, "price": buy_price,
                "amount": cost, "commission": commission, "stamp": 0,
                "net": -total_cost,
            })

        cash -= buy_value
        month_record["buy_total"] = round(buy_value, 2)
        month_record["cash_after"] = round(cash, 2)

        # ---- Step 5: 估值 ----
        portfolio_market_value = cash
        for code, h in holdings.items():
            current_price = prices.get((code, rebal_date), h["cost_price"])
            portfolio_market_value += h["shares"] * current_price

        forward_rets = {}
        for _, row in selected.iterrows():
            forward_rets[row["code"]] = row["forward_ret_1m"]

        projected_value = cash
        for code, h in holdings.items():
            ret = forward_rets.get(code, 0)
            current_price = prices.get((code, rebal_date), h["cost_price"])
            projected_value += h["shares"] * current_price * (1 + ret)

        month_record["market_value"] = round(portfolio_market_value, 2)
        month_record["projected_next"] = round(projected_value, 2)
        month_record["n_holdings"] = len(holdings)

        portfolio_values.append(month_record)
        monthly_detail.append(month_record)

        # ---- 打印本月操作 ----
        print(f"\n{'─' * 70}")
        print(f"  【调仓 #{i+1}】{rebal_date.strftime('%Y-%m-%d')}  |  调仓前现金: ¥{start_cash:,.0f}")
        print(f"{'─' * 70}")

        if month_record["sells"]:
            print(f"  ▶ 卖出 ({len(month_record['sells'])}只):")
            for s in month_record["sells"]:
                print(f"    {s['code']} {s['name']:8s}  {s['shares']}股 × ¥{s['price']:.2f}  "
                      f"回收 ¥{s['proceeds']:,.0f}  |  盈亏 {s['pnl']:+,.0f}")

        if month_record["holds"]:
            print(f"  ▶ 继续持有 ({len(month_record['holds'])}只):")
            for h in month_record["holds"]:
                chg = (h["now"] / h["cost"] - 1) * 100 if h["cost"] > 0 else 0
                print(f"    {h['code']} {h['name']:8s}  {h['shares']}股  成本¥{h['cost']:.2f}  "
                      f"现价¥{h['now']:.2f}  |  {chg:+.1f}%")

        print(f"  ▶ 买入 ({len(month_record['buys'])}只):")
        for b in month_record["buys"]:
            print(f"    {b['code']} {b['name']:8s} [{b['industry']:6s}]  "
                  f"{b['shares']}股 × ¥{b['price']:.2f}  "
                  f"花费 ¥{b['cost']:,.0f}  |  得分 {b['score']:.3f}")

        print(f"  {'─' * 50}")
        print(f"  卖出回收: ¥{month_record['sell_total']:,.0f}  |  "
              f"买入花费: ¥{month_record['buy_total']:,.0f}  |  "
              f"剩余现金: ¥{cash:,.0f}")
        print(f"  当前市值: ¥{portfolio_market_value:,.0f}  |  "
              f"下月预期: ¥{projected_value:,.0f}  |  "
              f"持仓 {len(holdings)} 只")

    # ============================================================
    # 4. 最终结算
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"  最终结算 ({final_date.strftime('%Y-%m-%d')})")
    print(f"{'=' * 70}")

    final_value = cash
    print(f"\n  持仓明细:")
    print(f"  {'代码':8s} {'股数':>6s} {'成本价':>8s} {'现价':>8s} {'市值':>10s} {'盈亏':>8s}")
    print(f"  {'-' * 55}")

    for code, h in sorted(holdings.items()):
        final_price = prices.get((code, final_date))
        if final_price is None:
            final_price = h["cost_price"]
        mkt_val = h["shares"] * final_price
        pnl = mkt_val - h["shares"] * h["cost_price"]
        final_value += mkt_val
        print(f"  {code:8s} {h['shares']:>6d} {h['cost_price']:>8.2f} {final_price:>8.2f} {mkt_val:>10,.0f} {pnl:>+8,.0f}")

    print(f"  {'-' * 55}")
    print(f"  现金: ¥{cash:,.0f}")
    print(f"  总市值: ¥{final_value:,.0f}")

    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"\n  初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print(f"  最终资金: ¥{final_value:,.0f}")
    print(f"  总收益: {total_return:+.1f}%")

    # 年化
    n_months = len(rebalance_dates)
    ann_ret = ((final_value / INITIAL_CAPITAL) ** (12 / n_months) - 1) * 100 if n_months > 0 else 0
    print(f"  年化收益: {ann_ret:+.1f}%")
    print(f"  模拟月数: {n_months}")

    # ---- 交易统计 ----
    if trades_log:
        trades_df = pd.DataFrame(trades_log)
        total_commission = trades_df["commission"].sum()
        total_stamp = trades_df["stamp"].sum()
        n_buys = (trades_df["action"] == "买入").sum()
        n_sells = (trades_df["action"] == "卖出").sum()
        print(f"\n  交易统计:")
        print(f"  买入 {n_buys} 笔, 卖出 {n_sells} 笔")
        print(f"  总佣金: ¥{total_commission:,.0f}, 总印花税: ¥{total_stamp:,.0f}")
        print(f"  交易成本合计: ¥{total_commission + total_stamp:,.0f}")

    # ---- 净值序列 ----
    pv_simple = [{
        "date": m["date"], "market_value": m["market_value"],
        "cash": m["cash_after"], "n_holdings": m["n_holdings"],
    } for m in monthly_detail]

    print(f"\n  逐月净值:")
    for m in monthly_detail:
        print(f"  {m['date'].strftime('%Y-%m')}  市值 ¥{m['market_value']:>10,.0f}  "
              f"现金 ¥{m['cash_after']:>8,.0f}  持仓 {m['n_holdings']} 只")

    # 保存详细记录
    import json
    detail_out = []
    for m in monthly_detail:
        d = {
            "date": m["date"].strftime("%Y-%m-%d"),
            "cash_before": m["cash_before"],
            "sell_total": m["sell_total"],
            "sells": m["sells"],
            "buy_total": m["buy_total"],
            "buys": [{"code": b["code"], "name": b["name"], "industry": b["industry"],
                      "shares": b["shares"], "price": b["price"],
                      "cost": b["cost"], "score": b["score"]} for b in m["buys"]],
            "holds": m["holds"],
            "cash_after": m["cash_after"],
            "market_value": m["market_value"],
            "n_holdings": m["n_holdings"],
        }
        detail_out.append(d)

    with open(ROOT / "data" / "paper_trade_detail.json", "w", encoding="utf-8") as f:
        json.dump(detail_out, f, ensure_ascii=False, indent=2)
    print(f"\n详细记录已保存: data/paper_trade_detail.json")

    # ---- 与基准对比 ----
    print(f"\n{'=' * 70}")
    print(f"  基准对比")
    print(f"{'=' * 70}")
    ft_sim = ft[ft["date"].isin(sim_dates)]
    # Top800等权
    bench_returns = []
    for d in sim_dates[:-1]:  # 不包括最后一天（没有 forward_ret）
        month = ft_sim[ft_sim["date"] == d].sort_values("factor_size", ascending=True).head(800)
        if len(month) > 0:
            bench_returns.append(month["forward_ret_1m"].mean())

    if bench_returns:
        bench_cum = (1 + pd.Series(bench_returns)).prod()
        bench_final = INITIAL_CAPITAL * bench_cum
        bench_ret = (bench_cum - 1) * 100
        print(f"  Top800等权基准: {bench_ret:+.1f}% (¥{bench_final:,.0f})")
        print(f"  策略超额: {total_return - bench_ret:+.1f}%")


if __name__ == "__main__":
    main()
