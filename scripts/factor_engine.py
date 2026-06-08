"""Factor computation engine - daily factors + financial merge + preprocessing"""

######### #?############### + ###### + ####""""



import sys

from pathlib import Path

import numpy as np

import pandas as pd

from tqdm import tqdm



ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "scripts"))



from factor_library import (

    FACTOR_DEFINITIONS, FACTOR_GROUPS, FactorSource,

    get_daily_factor_names, get_financial_factor_names, get_fin_mapping,

)





# ============================================================

# 1. ######################?# ============================================================



def compute_daily_factors(daily: pd.DataFrame) -> pd.DataFrame:

    """###################"""

    df = daily.copy()

    df = df.sort_values(["code", "date"])



    # --- ###### ---

    df["ret_1d"] = df.groupby("code")["close"].pct_change()



    # --- ##############?#??#?groupby().shift() ### transform(lambda) ##??---

    g_close = df.groupby("code")["close"]

    df["ret_5d"] = df["close"] / g_close.shift(5) - 1

    df["ret_1m"] = df["close"] / g_close.shift(21) - 1

    df["ret_3m"] = df["close"] / g_close.shift(63) - 1

    df["ret_12m"] = df["close"] / g_close.shift(252) - 1



    # ###### 12-1#?    df["momentum_12m1m"] = (1 + df["ret_12m"]) / (1 + df["ret_1m"]) - 1

    df["factor_momentum_12m1m"] = df["momentum_12m1m"]



    # ###### 1#?/ 3###############AI/####?    df["factor_momentum_1m"] = df["ret_1m"]

    df["factor_momentum_3m"] = df["ret_3m"]



    # --- ####?---

    df["vol_20d"] = df.groupby("code")["ret_1d"].rolling(20).std().droplevel(0)

    df["vol_60d"] = df.groupby("code")["ret_1d"].rolling(60).std().droplevel(0)

    df["factor_vol_20d"] = -df["vol_20d"]

    df["factor_vol_60d"] = -df["vol_60d"]



    # --- ####?---

    if "turnover" in df.columns:

        df["turnover_20d"] = df.groupby("code")["turnover"].rolling(20).mean().droplevel(0)

        df["turnover_5d"] = df.groupby("code")["turnover"].rolling(5).mean().droplevel(0)

        df["factor_turnover"] = -df["turnover_20d"]

        df["factor_abnormal_turn"] = -(df["turnover_5d"] / df["turnover_20d"] - 1)



    if "amount" in df.columns:

        df["amount_20d"] = df.groupby("code")["amount"].rolling(20).mean().droplevel(0)

        df["factor_size"] = -np.log(df["amount_20d"])



    # --- ##?#####1/ratio#?---

    for col, out_col in [("pe_ttm", "factor_ep"), ("pb_mrq", "factor_bp"),

                          ("ps_ttm", "factor_sp"), ("pcf_ttm", "factor_cfp")]:

        if col in df.columns:

            valid = df[col].copy()

            valid[valid <= 0] = np.nan

            df[out_col] = 1.0 / valid



    # --- #?####?---

    df["factor_reversal"] = -df["ret_5d"]



    if "pe_ttm" in df.columns:

        g_pe = df.groupby("code")["pe_ttm"]

        df["pe_change_1m"] = df["pe_ttm"] / g_pe.shift(21) - 1

        df["factor_pe_momentum"] = -df["pe_change_1m"]



    return df





# ============================================================

# 2. #######?# ============================================================



def winsorize_mad(series: pd.Series, n_mad: float = 5) -> pd.Series:

    """MAD #####?"""

    median = series.median()

    mad = (series - median).abs().median()

    if mad == 0 or pd.isna(mad):

        return series

    return series.clip(lower=median - n_mad * mad, upper=median + n_mad * mad)





def preprocess_cross_section(cross: pd.DataFrame,

                             factor_cols: list[str],

                             neutralize: bool = False) -> pd.DataFrame:

    """



    1. MAD ####?    2. Z-score ####?    3. ##?#####+#####?##

    """

    result = cross.copy()



    for col in factor_cols:

        if col not in result.columns or result[col].isna().all():

            continue



        # Step 1: MAD ####?        result[col] = winsorize_mad(result[col].copy())



        # Step 2: Z-score

        mean = result[col].mean()

        std = result[col].std()

        if std > 0:

            result[col] = (result[col] - mean) / std

        else:

            result[col] = 0



    # Step 3: ###+#####?##

    if neutralize and "industry" in result.columns and "factor_size" in result.columns:

        print("  ###+#####?##...")

        for col in tqdm(factor_cols, desc="##?##"):

            valid = result[col].notna() & result["factor_size"].notna()

            if valid.sum() < 10:

                continue



            y = result.loc[valid, col].values

            ind_dummies = pd.get_dummies(

                result.loc[valid, "industry"], drop_first=True

            ).astype(float)

            X = np.column_stack([

                result.loc[valid, "factor_size"].values,

                ind_dummies.values,

            ])

            X = np.hstack([np.ones((X.shape[0], 1)), X])



            try:

                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

                residuals = y - X @ beta

                result.loc[valid, col] = residuals

            except Exception as e:

                print(f"[factor_engine] neutralization failed for {col}: {e}")



    return result





# ============================================================

# 3. #############?##########?# ============================================================



def apply_financial_factors(fin_df: pd.DataFrame,

                            fin_map: dict[str, str]) -> pd.DataFrame:

    """##################### FactorDefinition ######"""

    from factor_library import FACTOR_DEFINITIONS

    result = fin_df.copy()

    for src, dst in fin_map.items():

        if src in result.columns:

            result[dst] = result[src]

            fdef = FACTOR_DEFINITIONS.get(dst)

            if fdef and fdef.needs_sign_flip:

                result[dst] = -result[src]

    return result





# ============================================================

# 4. ##################/IC####?# ============================================================



def build_monthly_factor_table(daily: pd.DataFrame,

                               fin: pd.DataFrame,

                               start_date: str = "2021-01-01") -> pd.DataFrame:

    """

    ###################?+ forward_ret_1m#?    ###IC##########?    """

    print("##########?..")



    # ######

    df = compute_daily_factors(daily)



    # #########

    from data_orchestrator import align_financials_to_dates

    fin_map = get_fin_mapping()

    fin_daily = align_financials_to_dates(df["date"], fin, fin_map)



    # #######?#?################?    fin_daily = apply_financial_factors(fin_daily, fin_map)



    available_fin_factors = [dst for src, dst in fin_map.items()

                             if src in fin_daily.columns]

    df = df.merge(

        fin_daily[["date", "code"] + available_fin_factors],

        on=["date", "code"], how="left")



    # ############

    for col in available_fin_factors:

        df[col] = df.groupby("code")[col].transform(lambda x: x.ffill(limit=4))  # max 4 periods (~1 year)



    # #######?    df["year_month"] = df["date"].dt.to_period("M")

    monthly_dates = sorted(df.groupby("year_month")["date"].max().tolist())

    month_end = df[df["date"].isin(monthly_dates)].copy()



    # forward_ret_1m

    month_end = month_end.sort_values(["code", "date"])

    month_end["forward_close"] = month_end.groupby("code")["close"].shift(-1)

    month_end["forward_ret_1m"] = month_end["forward_close"] / month_end["close"] - 1



    month_end = month_end[month_end["date"] >= pd.Timestamp(start_date)]



    # ####?    all_factor_cols = [c for c in month_end.columns if c.startswith("factor_")]

    out_cols = ["date", "code"] + all_factor_cols + ["forward_ret_1m"]

    result = month_end[out_cols].dropna(subset=["forward_ret_1m"]).copy()



    # #######?    for col in tqdm(all_factor_cols, desc="####?):

    result[col] = result.groupby("date")[col].transform(

            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)



    result = result.sort_values(["date", "code"]).reset_index(drop=True)

    n_factors = len(all_factor_cols)

    n_months = result["date"].nunique()

    n_stocks = result["code"].nunique()

#    print(f"  #######? {n_months} ###, {n_stocks} ####? {n_factors} ####?\")



    return result





# ============================================================

# 5. #########

# ============================================================



def score_cross_section(cross: pd.DataFrame,

                        factor_weights: dict[str, float],

                        ic_report: pd.DataFrame = None) -> pd.DataFrame:

    """





    cross: ##########?DataFrame

    factor_weights: {factor_name: weight}

    ic_report: IC##################



    ###: #?'score' ### DataFrame############

    """

    result = cross.copy()

    result["score"] = 0.0

    weight_sum = 0.0



    for f, weight in factor_weights.items():

        if f not in result.columns:

            continue



        vals = result[f].fillna(0).values



        # ######

        sign = 1

        if ic_report is not None and "factor" in ic_report.columns:

            row = ic_report[ic_report["factor"] == f]

            if len(row) > 0 and row["ic_mean"].values[0] < 0:

                sign = -1

        else:

            lookup = f.replace("_raw", "") if f.endswith("_raw") else f

            fdef = FACTOR_DEFINITIONS.get(lookup)

            if fdef and fdef.direction.value == -1:

                sign = -1



        result["score"] += vals * sign * weight

        weight_sum += abs(weight)



    if weight_sum > 0:

        result["score"] /= weight_sum  # ##########?#?

    result = result.sort_values("score", ascending=False)

    return result

