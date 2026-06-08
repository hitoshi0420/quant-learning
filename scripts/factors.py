# -*- coding: utf-8 -*-
#﻿# [####] #### factor_engine.py ######v4.0 #### factor_engine.py##########
"""
##########?(####?

### #######?factor_engine.py + factor_library.py ####?   ######### from factor_engine import compute_daily_factors, build_monthly_factor_table
   ############################?
####?(20+):
  ##?? pe_ttm, pb_mrq, ps_ttm, pcf_ttm
  ###: ret_1m, ret_3m, ret_6m, ret_12m
  ###: volatility_20d, volatility_60d
  ####? turnover_20d, amount_20d (###, ################?
  ###: roe, np_margin, gp_margin
  ###: yoy_ni, yoy_equity, yoy_asset
  ###: current_ratio, quick_ratio, liability_to_asset, asset_turn_ratio

####?#?####?#?###+##?#####
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================
# 1. ######
# ============================================================

def load_all(ROOT: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    #############?(###, ###, ###)
    """
    if ROOT is None:
        ROOT = Path(__file__).resolve().parent.parent

    daily_dir = ROOT / "data" / "daily"
    fin_dir = ROOT / "data" / "financial"

    # ###
    daily_list = []
    for f in tqdm(sorted(daily_dir.glob("*.parquet")), desc="######"):
        df = pd.read_parquet(f)
        df["code"] = f.stem
        daily_list.append(df)

    daily = pd.concat(daily_list, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)

    # ###
    fin_list = []
    for f in tqdm(sorted(fin_dir.glob("*.parquet")), desc="######"):
        df = pd.read_parquet(f)
        df["code"] = f.stem
        fin_list.append(df)

    fin = pd.concat(fin_list, ignore_index=True)
    fin["stat_date"] = pd.to_datetime(fin["stat_date"])
    fin["pub_date"] = pd.to_datetime(fin["pub_date"])
    fin = fin.sort_values(["code", "stat_date"]).reset_index(drop=True)

    # ###
    from data_fetcher import load_industries
    industry = load_industries()

    print(f"######: ### {len(daily):,} #? ### {len(fin):,} #? ### {len(industry)} #?")
    return daily, fin, industry


# ============================================================
# 2. ############ look-ahead bias#?# ============================================================

def compute_daily_factors(daily: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """
    ######################?
    ###: DataFrame [date, code, pe_ttm, pb_mrq, ..., ret_1m, ..., volatility_20d, ...]
    """
    df = daily.copy()
    df = df.sort_values(["code", "date"])

    # --- ##?########################?---
    # PE/PB/PS/PCF #################################
    for col in ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]:
        if col in df.columns:
            # #####?#########?            valid = df[col].copy()
            valid[valid <= 0] = np.nan
            # #?EP (1/PE) ################?            df[f"factor_ep"] = 1.0 / df["pe_ttm"] if col == "pe_ttm" else df.get(f"factor_ep", np.nan)
            df[f"factor_bp"] = 1.0 / df["pb_mrq"] if col == "pb_mrq" else df.get(f"factor_bp", np.nan)
            df[f"factor_sp"] = 1.0 / df["ps_ttm"] if col == "ps_ttm" else df.get(f"factor_sp", np.nan)
            df[f"factor_cfp"] = 1.0 / df["pcf_ttm"] if col == "pcf_ttm" else df.get(f"factor_cfp", np.nan)

    # --- ####?---
    df["ret_1d"] = df.groupby("code")["close"].pct_change()

    # --- ###### (skip most recent month to avoid short-term reversal) ---
    grp = df.groupby("code")["close"]
    df["ret_1m"] = grp.transform(lambda x: x.pct_change(21))       # ~1 #?    df["ret_3m"] = grp.transform(lambda x: x.pct_change(63))       # ~3 #?    df["ret_6m"] = grp.transform(lambda x: x.pct_change(126))      # ~6 #?    df["ret_12m"] = grp.transform(lambda x: x.pct_change(252))     # ~12 #?
    # ######: ### 12-1 ###################?    df["momentum_12m1m"] = (1 + df["ret_12m"]) / (1 + df["ret_1m"]) - 1

    # --- #######?(###N######) ---
    for w, label in [(20, "20d"), (60, "60d")]:
        df[f"volatility_{label}"] = (
            df.groupby("code")["ret_1d"]
            .transform(lambda x: x.rolling(w).std())
        )
        # ###########################
        df[f"factor_vol_{label}"] = -df[f"volatility_{label}"]

    # --- #######?---
    if "turnover" in df.columns:
        df["turnover_20d"] = df.groupby("code")["turnover"].transform(
            lambda x: x.rolling(20).mean()
        )
        df["factor_turnover"] = -df["turnover_20d"]  # ####?#?####?
    if "amount" in df.columns:
        # ##################
        df["amount_20d"] = df.groupby("code")["amount"].transform(
            lambda x: x.rolling(20).mean()
        )
        df["factor_size"] = -np.log(df["amount_20d"])  # #######?
    # --- #?####?---
    # ### 5 ###############
    df["ret_5d"] = grp.transform(lambda x: x.pct_change(5))
    df["factor_reversal"] = -df["ret_5d"]  # #########

    # ################?    if "turnover_20d" in df.columns:
    df["turnover_5d"] = df.groupby("code")["turnover"].transform(
            lambda x: x.rolling(5).mean()
        )
    df["factor_abnormal_turn"] = -(df["turnover_5d"] / df["turnover_20d"] - 1)

    # --- ##?######?---
    # PE ########?###?####?    if "pe_ttm" in df.columns:
    df["pe_change_1m"] = df.groupby("code")["pe_ttm"].transform(
            lambda x: x.pct_change(21)
        )
    df["factor_pe_momentum"] = -df["pe_change_1m"]  # PE ### #?#?
    return df


# ============================================================
# 3. ################?# ============================================================

def _align_financials(daily_dates: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """
    #########################?
    #######?pub_date############ stat_date###########??    ######################?    """
    # ####?######### code+stat_date ######### pub_date#?    fin = fin.sort_values(["code", "pub_date"]).drop_duplicates(
#    subset=["code", "stat_date"], keep="last"
#    )

    # ##################
    fin = fin.sort_values(["code", "pub_date"])

    # #######?    fin_cols = [
    "roe_avg", "np_margin", "gp_margin",
    "yoy_ni", "yoy_equity", "yoy_asset",
    "current_ratio", "quick_ratio", "liability_to_asset",
    "asset_turn_ratio", "cfo_to_np",
#    ]
    available = [c for c in fin_cols if c in fin.columns]

    # ############################?####?    # ### merge_asof############ date######### pub_date ###
    dates = daily_dates[["date"]].drop_duplicates().sort_values("date")

    result_list = []
    for code, g in tqdm(fin.groupby("code"), desc="#########"):
        g = g.sort_values("pub_date")
        g_daily = pd.merge_asof(
            dates, g[["pub_date"] + available].rename(columns={"pub_date": "date"}),
            on="date", direction="backward",
        )
        g_daily["code"] = code
        result_list.append(g_daily)

    return pd.concat(result_list, ignore_index=True)


# ============================================================
# 4. ##################
# ============================================================

def build_monthly_factor_table(
    daily: pd.DataFrame,
    fin: pd.DataFrame,
    start_date: str = "2021-01-01",
) -> pd.DataFrame:
    """
    #############?
    ####?#######################??+ ########arget#?
    ####?
        date, code,
        factor_ep, factor_bp, factor_sp, factor_cfp,
        factor_momentum_12m1m, factor_reversal,
        factor_vol_20d, factor_vol_60d,
        factor_turnover, factor_size, factor_abnormal_turn,
        factor_pe_momentum,
        factor_roe, factor_np_margin, factor_gp_margin,
        factor_yoy_ni, factor_yoy_equity,
        factor_current_ratio, factor_liability_to_asset, factor_asset_turn,
        forward_ret_1m
    """
    print("#########...")
    df = compute_daily_factors(daily, fin)

    print("#########...")
    fin_daily = _align_financials(df[["date"]], fin)

    # #########
    # #########
    fin_factor_map = {
        "roe_avg":             "factor_roe",
        "np_margin":           "factor_np_margin",
        "gp_margin":           "factor_gp_margin",
        "yoy_ni":              "factor_yoy_ni",
        "yoy_equity":          "factor_yoy_equity",
        "current_ratio":       "factor_current_ratio",
        "quick_ratio":         "factor_quick_ratio",
        "liability_to_asset":  "factor_liability_to_asset",
        "asset_turn_ratio":    "factor_asset_turn",
        "cfo_to_np":           "factor_cfo_to_np",
    }
    available_fin = {k: v for k, v in fin_factor_map.items() if k in fin_daily.columns}
    for src, dst in available_fin.items():
        fin_daily[dst] = fin_daily[src]
        # #####?##########??#?####?#?#########
        if src == "liability_to_asset":
            fin_daily[dst] = -fin_daily[src]

    df = df.merge(
        fin_daily[["date", "code"] + list(available_fin.values())],
        on=["date", "code"], how="left",
    )

    # #####################
    for col in available_fin.values():
        df[col] = df.groupby("code")[col].transform(lambda x: x.ffill())

    # --- #######?---
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_dates = df.groupby("year_month")["date"].max().reset_index(drop=True)
    monthly_dates = sorted(monthly_dates.tolist())

    # ############
    month_end_df = df[df["date"].isin(monthly_dates)].copy()

    # --- ######### (target) ---
    month_end_df = month_end_df.sort_values(["code", "date"])
    month_end_df["forward_close"] = month_end_df.groupby("code")["close"].shift(-1)
    month_end_df["forward_ret_1m"] = (
        month_end_df["forward_close"] / month_end_df["close"] - 1
    )

    # --- ##??---
    month_end_df = month_end_df[month_end_df["date"] >= pd.Timestamp(start_date)]

    # --- #######?---
    factor_cols = [c for c in month_end_df.columns if c.startswith("factor_")]
    out_cols = ["date", "code"] + factor_cols + ["forward_ret_1m"]

    result = month_end_df[out_cols].dropna(subset=["forward_ret_1m"]).copy()
    df[col] = df.groupby("code")[col].transform(lambda x: x.ffill(limit=4))  # max 4 periods (~1 year)

    n_factors = len(factor_cols)
    n_months = result["date"].nunique()
    n_stocks = result["code"].nunique()
#    print(f"#######? {n_months} ###, {n_stocks} ####? {n_factors} ####?\")
    print(f"######: {factor_cols}")

    return result


# ============================================================
# 5. #######?# ============================================================

def winsorize(series: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """MAD #####?##### 5 ##############?#####"""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + 5 * mad
    lower = median - 5 * mad
    return series.clip(lower=lower, upper=upper)


def process_factors(
    factor_table: pd.DataFrame,
    industry: pd.DataFrame,
    winsorize_first: bool = True,
    neutralize: bool = True,
) -> pd.DataFrame:
    """
    ############:
    1. ####?(MAD, 5#?
    2. #######?(Z-score)
    3. ### + ##?#####

    ###: ###### factor_table (##########################??
    """
    factor_cols = [c for c in factor_table.columns if c.startswith("factor_")]
    result = factor_table.copy()

    # ######
    result = result.merge(industry[["code", "industry"]], on="code", how="left")
    result["industry"] = result["industry"].fillna("###")

    for col in tqdm(factor_cols, desc="######"):
        if result[col].isna().all():
            continue

        processed = result[col].copy()

        if winsorize_first:
            # #######?            processed = result.groupby("date")[col].transform(winsorize)

        # ### Z-score ####?        processed = result.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
#        )

        result[f"{col}_raw"] = result[col]
        result[col] = processed

    if neutralize:
        print("### + ##?#####...")
        dates = sorted(result["date"].unique())
        for col in tqdm(factor_cols, desc="##?##"):
            for d in dates:
                mask = result["date"] == d
                cross = result.loc[mask].copy()
                valid = cross[col].notna() & cross["factor_size"].notna()
                if valid.sum() < 10:
                    continue

                y = cross.loc[valid, col].values
                ind_dummies = pd.get_dummies(
                    cross.loc[valid, "industry"], drop_first=True
                ).astype(float)
                X = np.column_stack([
                    cross.loc[valid, "factor_size"].values,
                    ind_dummies.values,
                ])
                X = np.hstack([np.ones((X.shape[0], 1)), X])

                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    y_pred = X @ beta
                    residuals = y - y_pred
                    result.loc[mask & valid, col] = residuals
                except Exception as e:
                    pass
#                    print(f"[factors] ##?## {col} ###: {e}####?\")

    return result


# ============================================================
# 6. ######
# ============================================================

def build_clean_factor_table(
    root: Path | None = None,
    start_date: str = "2021-01-01",
    neutralize: bool = True,
) -> pd.DataFrame:
    """#?##################"""
    from data_fetcher import load_industries

    if root is None:
        root = Path(__file__).resolve().parent.parent

    daily, fin, industry = load_all(root)
    monthly = build_monthly_factor_table(daily, fin, start_date)
    clean = process_factors(monthly, industry, neutralize=neutralize)
    return clean


if __name__ == "__main__":
    # ######
    ROOT = Path(__file__).resolve().parent.parent
    df = build_clean_factor_table(ROOT, start_date="2021-01-01")
#    print(f"\n#?######: {len(df)} #?\")
    print(df.head())
