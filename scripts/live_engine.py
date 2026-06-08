"""
Live Prediction Engine v4.0 - Multi-strategy parallel + style diversification + dynamic IC
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from factor_library import (
    FACTOR_DEFINITIONS, FACTOR_GROUPS, MAX_GROUP_WEIGHT,
    MAX_SINGLE_FACTOR_WEIGHT, FactorDirection,
    get_daily_factor_names, get_financial_factor_names,
)
from portfolio_builder import PortfolioMethod, build_portfolio
from risk_manager import RiskParams, apply_risk_controls
from factor_library import STRATEGY_CLUSTERS


class LiveEngine:
    """Live prediction engine v4.0 - multi-strategy parallel"""

    def __init__(self):
        self.daily = None
        self.fin = None
        self.industry_df = None
        self.name_map = {}
        self.ind_map = {}
        self.factor_table = None
        self.target_date = None
        self.cross = None
        self.clusters = {}
        self.portfolio = None

    # ============================================================
    # 1. Data preparation
    # ============================================================

    def load_data(self, lookback_days: int = None):
        """Load all necessary data"""
        if lookback_days is None:
            from config import LIQUIDITY_LOOKBACK_DAYS
            lookback_days = LIQUIDITY_LOOKBACK_DAYS
        from data_orchestrator import (
            load_daily_recent, load_financial_data,
            load_stock_names, load_industries, preflight_check,
        )

        ready, issues = preflight_check()
        if not ready:
            print("Warning: data preflight check found issues, continuing...")
            for issue in issues:
                print(f"  - {issue}")

        self.daily = load_daily_recent(lookback_days)
        self.name_map = load_stock_names()
        self.industry_df = load_industries()
        self.ind_map = dict(zip(self.industry_df["code"], self.industry_df["industry"]))

        try:
            self.fin = load_financial_data()
        except Exception as e:
            print(f"Financial data load failed: {e}, using daily factors only")
            self.fin = None

        from data_orchestrator import load_cached_factor_table
        self.factor_table = load_cached_factor_table()

    # ============================================================
    # 2. Factor computation
    # ============================================================

    def compute_factors(self, target_date: Optional[pd.Timestamp] = None):
        """Compute full factor cross-section for target date"""
        from factor_engine import compute_daily_factors, apply_financial_factors
        from data_orchestrator import get_latest_full_date
        from factor_library import get_fin_mapping

        if target_date is None:
            target_date = get_latest_full_date(self.daily)

        self.target_date = target_date
        print(f"Target date: {target_date.strftime('%Y-%m-%d')}")

        # Daily factors
        df = compute_daily_factors(self.daily)
        cross = df[df["date"] == target_date].copy()
        print(f"  Daily cross-section: {len(cross)} stocks")

        # Merge industry
        if self.industry_df is not None:
            cross = cross.merge(self.industry_df[["code", "industry"]],
                                on="code", how="left")
            cross["industry"] = cross["industry"].fillna("Unknown")

        # Merge financial factors
        if self.fin is not None and len(self.fin) > 0:
            fin_map = get_fin_mapping()
            fin_recent = self.fin[self.fin["pub_date"] <= target_date]
            if len(fin_recent) > 0:
                fin_latest = fin_recent.sort_values("pub_date").groupby("code").last().reset_index()
                fin_latest = apply_financial_factors(fin_latest, fin_map)
                available_fin = [dst for src, dst in fin_map.items()
                                 if src in fin_latest.columns]
                if available_fin:
                    cross = cross.merge(
                        fin_latest[["code"] + available_fin],
                        on="code", how="left")
                    n_with_fin = cross[available_fin[0]].notna().sum()
                    print(f"  Financial factors: {len(available_fin)}, {n_with_fin} stocks have financial data")

        # Liquidity filter (Top 800 by avg amount)
        if "amount_20d" in cross.columns:
            cross = cross.dropna(subset=["amount_20d"])
            cross = cross.sort_values("amount_20d", ascending=False).head(800)
            print(f"  Liquidity filter (Top800): {len(cross)} stocks")

        self.cross = cross

    # ============================================================
    # 3. IC estimation + factor selection (per cluster)
    # ============================================================

    def estimate_ic(self, max_train_date: Optional[pd.Timestamp] = None):
        """Run IC analysis independently for each strategy cluster"""
        from ic_estimator import run_ic_pipeline
        from config import IC_LOOKBACK_MONTHS

        if self.factor_table is None or self.factor_table["date"].nunique() < IC_LOOKBACK_MONTHS:
            n_months = self.factor_table["date"].nunique() if self.factor_table is not None else 0
            print(f"Insufficient history ({n_months} months, need {IC_LOOKBACK_MONTHS}), using default equal-weight factors")
            self._fallback_factors()
            return

        # Determine training window cutoff
        if max_train_date is None and self.target_date is not None:
            max_train_date = self.target_date - pd.offsets.MonthBegin(1)
        if max_train_date is not None:
            train_table = self.factor_table[self.factor_table["date"] <= max_train_date].copy()
            if train_table["date"].nunique() < 6:
                print(f"Training window insufficient (cutoff {max_train_date.date()}), using full history")
                train_table = self.factor_table.copy()
        else:
            train_table = self.factor_table.copy()

        # IC decay diagnosis (full universe, once per factor)
        decay_info = self._check_ic_decay(train_table)

        print("Multi-strategy IC analysis + factor selection..."
              + (f" (training window <= {max_train_date.date()})" if max_train_date is not None else ""))

        for cluster_name, config in STRATEGY_CLUSTERS.items():
            print(f"\n-- {{cluster_name}} ({{config[dq]style[dq]}}) --")
            print(f"  {{config[dq]description[dq]}}")
            print(f"  {config['description']}")

            # Aggregate factor list for this cluster
            group_factors = []
            for g in config["factor_groups"]:
                if g in FACTOR_GROUPS:
                    group_factors.extend(FACTOR_GROUPS[g])

            factor_cols = [c for c in group_factors
                           if c in train_table.columns and c != "factor_size"]

            if len(factor_cols) < 3:
                print(f"  {cluster_name}: insufficient factors ({len(factor_cols)}), skipping cluster")
                continue

            filtered_cols = ["date", "code", "forward_ret_1m"] + factor_cols
            filtered_table = train_table[filtered_cols].copy()

            try:
                from config import MIN_IC_IR_THRESHOLD
                ic_report, selected_factors, factor_weights = run_ic_pipeline(
                    filtered_table,
                    min_ic_ir=MIN_IC_IR_THRESHOLD,
                    max_group_weight=None,
                )
            except Exception as e:
                print(f"  IC analysis failed: {e}, using equal weight")
                selected_factors = factor_cols
                factor_weights = {f: 1.0 / len(factor_cols) for f in factor_cols}
                ic_report = pd.DataFrame(columns=["factor", "ic_mean", "ic_ir"])

            # Apply IC decay penalty
            if decay_info:
                factor_weights, selected_factors = self._apply_decay_penalty(
                    factor_weights, selected_factors, decay_info, cluster_name)

            self.clusters[cluster_name] = {
                "ic_report": ic_report,
                "selected_factors": selected_factors,
                "factor_weights": factor_weights,
                "factor_groups": config["factor_groups"],
                "pick_count": config["pick_count"],
                "style": config["style"],
                "description": config["description"],
                "decay_info": decay_info,
            }

        if not self.clusters:
            print("All clusters failed to initialize, using default equal-weight factors")
            self._fallback_factors()

    def _check_ic_decay(self, train_table: pd.DataFrame) -> dict:
        """IC decay diagnosis for all factors"""
        try:
            from ic_decay import extract_ic_series, ic_decay_test

            ic_df = extract_ic_series(train_table)
            if len(ic_df) < 12:
                print("    IC series < 12 periods, skipping decay diagnosis")
                return {}

            decay_report = ic_decay_test(ic_df)
            if len(decay_report) == 0:
                return {}

            severe = decay_report[decay_report["decay_score"] >= 3]
            if len(severe) > 0:
                print(f"  Severe decay factors ({len(severe)}):")
                for _, r in severe.iterrows():
                    factor_full = f"factor_{r['factor']}"
                    print(f"    {factor_full:28s} IC={r['ic_mean']:+.4f} "
                          f"recent={r['recent_ic']:+.4f} beta={r['trend_beta']*100:+.3f}%/month")

            mild = decay_report[(decay_report["decay_score"] >= 1) & (decay_report["decay_score"] <= 2)]
            if len(mild) > 0:
                print(f"  Mild decay factors ({len(mild)}):")
                for _, r in mild.head(6).iterrows():
                    factor_full = f"factor_{r['factor']}"
                    print(f"    {factor_full:28s} IC={r['ic_mean']:+.4f} "
                          f"recent={r['recent_ic']:+.4f} score={r['decay_score']}")

            info = {}
            for _, r in decay_report.iterrows():
                info[f"factor_{r['factor']}"] = {
                    "decay_score": r["decay_score"],
                    "ic_mean": r["ic_mean"],
                    "recent_ic": r["recent_ic"],
                    "trend_beta": r["trend_beta"],
                    "trend_p": r["trend_p"],
                }
            return info

        except Exception as e:
            print(f"  IC decay diagnosis failed: {e}")
            return {}

    def _apply_decay_penalty(self, factor_weights: dict, selected_factors: list,
                             decay_info: dict, cluster_name: str) -> tuple:
        """Apply weight penalty for severely decaying factors"""
        excluded = []
        penalized = []

        for f in list(factor_weights.keys()):
            di = decay_info.get(f)
            if di is None:
                continue
            score = di["decay_score"]

            if score >= 4:
                excluded.append((f, score))
                factor_weights.pop(f, None)
                if f in selected_factors:
                    selected_factors.remove(f)
            elif score == 3:
                penalized.append((f, score, 0.30))
                factor_weights[f] *= 0.30
            elif score == 2:
                penalized.append((f, score, 0.70))
                factor_weights[f] *= 0.70

        if excluded:
            names = [f[0].replace("factor_", "") for f in excluded]
            print(f"  [{cluster_name}] Excluded decayed factors: {', '.join(names)}")
        if penalized:
            for f, score, mult in penalized:
                short = f.replace("factor_", "")
                print(f"  [{cluster_name}] Downweighted {short} (decay={score}, x{mult:.0%})")

        w_sum = sum(factor_weights.values())
        if w_sum > 0 and len(factor_weights) > 0:
            factor_weights = {k: v / w_sum for k, v in factor_weights.items()}

        return factor_weights, selected_factors

    def _fallback_factors(self):
        """When no historical data available, use daily factors equal-weight"""
        daily_factors = get_daily_factor_names()
        available = [f for f in daily_factors if f in self.cross.columns]

        for cluster_name, config in STRATEGY_CLUSTERS.items():
            group_factors = []
            for g in config["factor_groups"]:
                if g in FACTOR_GROUPS:
                    group_factors.extend(FACTOR_GROUPS[g])
            cluster_factors = [f for f in group_factors if f in available]
            if not cluster_factors:
                cluster_factors = available[:5]
            equal_w = 1.0 / max(len(cluster_factors), 1)
            self.clusters[cluster_name] = {
                "ic_report": pd.DataFrame(columns=["factor", "ic_mean", "ic_ir"]),
                "selected_factors": cluster_factors,
                "factor_weights": {f: equal_w for f in cluster_factors},
                "factor_groups": config["factor_groups"],
                "pick_count": config["pick_count"],
                "style": config["style"],
                "description": config["description"],
            }
        print(f"  Default factors: {len(available)} daily factors, equal-weight assigned")

    # ============================================================
    # 4. Preprocessing + Scoring (per cluster)
    # ============================================================

    def score(self, neutralize: bool = False):
        """Independent preprocessing + scoring for each cluster"""
        from factor_engine import preprocess_cross_section, score_cross_section

        for cluster_name, cluster_data in self.clusters.items():
            cross_copy = self.cross.copy()
            proc_factors = [f for f in cluster_data["selected_factors"]
                            if f in cross_copy.columns]

            if len(proc_factors) < 2:
                print(f"  {cluster_name}: insufficient valid factors, skipping")
                cluster_data["cross"] = cross_copy
                cluster_data["cross"]["score"] = 0.0
                continue

            # Preprocess (MAD + Z-score, no industry neutralization to preserve style purity)
            cross_copy = preprocess_cross_section(cross_copy, proc_factors,
                                                  neutralize=False)

            # Score
            cross_copy = score_cross_section(cross_copy, cluster_data["factor_weights"],
                                             cluster_data["ic_report"])

            cluster_data["cross"] = cross_copy
            top_scores = cross_copy["score"].head(5).values.round(3)
            print(f"  {cluster_name}: Top5 scores {top_scores}")

    # ============================================================
    # 5. Portfolio construction (multi-cluster ensemble + risk control)
    # ============================================================

    def build(self, method: PortfolioMethod = PortfolioMethod.ICIR_WEIGHTED):
        """Take top-N from each cluster, merge, deduplicate, apply risk controls"""
        all_picks = []
        cluster_summary = []

        for cluster_name, cluster_data in self.clusters.items():
            if "cross" not in cluster_data or cluster_data["cross"].empty:
                continue

            n_picks = cluster_data.get("pick_count", 3)
            cross = cluster_data["cross"]

            top = cross.head(n_picks).copy()
            top["cluster"] = cluster_name
            top["cluster_style"] = cluster_data.get("style", "")
            all_picks.append(top)

            cluster_summary.append({
                "cluster": cluster_name,
                "n_picks": len(top),
                "top_stocks": top["code"].tolist(),
                "top_names": [self.name_map.get(c, "") for c in top["code"]],
            })

        if not all_picks:
            print("No valid recommendations")
            self.portfolio = pd.DataFrame()
            return

        combined = pd.concat(all_picks, ignore_index=True)

        # Deduplicate (same stock picked by multiple clusters, keep first = higher rank)
        combined = combined.drop_duplicates(subset=["code"], keep="first")

        # Initial weights by score
        if "score" in combined.columns:
            min_s = combined["score"].min()
            shifted = combined["score"] - min_s + 0.1
            combined["weight"] = shifted / shifted.sum()
        else:
            combined["weight"] = 1.0 / len(combined)

        # Risk controls
        params = RiskParams()
        combined = apply_risk_controls(
            combined, params=params,
            industry_map=self.ind_map,
        )

        self.portfolio = combined

        # Industry distribution
        industries = [self.ind_map.get(c, "Unknown") for c in combined["code"]]
        ind_dist = Counter(industries)
        max_ind_pct = max(ind_dist.values()) / max(len(combined), 1)

        print(f"\n  Portfolio: {len(combined)} stocks, max industry concentration: {max_ind_pct:.0%}")

        if "cluster" in combined.columns:
            for cname in combined["cluster"].unique():
                count = (combined["cluster"] == cname).sum()
                print(f"    {cname}: {count} stocks")

        return combined

    # ============================================================
    # 5a. Factor attribution - generate recommendation reasons
    # ============================================================

    def _analyze_pick_reasons(self) -> dict:
        """Analyze factor contributions for each recommended stock"""
        from factor_library import FACTOR_DEFINITIONS

        reasons = {}
        portfolio = self.portfolio
        if portfolio is None or len(portfolio) == 0:
            return reasons

        for _, row in portfolio.iterrows():
            code = row["code"]
            cluster_name = row.get("cluster", "")
            cluster_data = self.clusters.get(cluster_name, {})

            cross = cluster_data.get("cross")
            selected_factors = cluster_data.get("selected_factors", [])
            factor_weights = cluster_data.get("factor_weights", {})

            contributions = []
            stock_row = None
            if cross is not None and "code" in cross.columns:
                mask = cross["code"] == code
                if mask.any():
                    stock_row = cross[mask].iloc[0]

            if stock_row is not None:
                for f in selected_factors:
                    if f not in stock_row.index:
                        continue
                    z = float(stock_row[f])
                    if pd.isna(z):
                        continue
                    w = factor_weights.get(f, 0)
                    contrib = abs(z * w)
                    look_f = f.replace("_raw", "") if f.endswith("_raw") else f
                    fdef = FACTOR_DEFINITIONS.get(look_f)
                    display = fdef.display_name if fdef else f
                    group = fdef.group if fdef else "unknown"
                    direction = "favorable" if ((z > 0 and fdef.direction.value == 1) or
                                               (z < 0 and fdef.direction.value == -1)) else "unfavorable"
                    contributions.append({
                        "factor": f,
                        "name": display,
                        "group": group,
                        "z_score": round(z, 3),
                        "weight": round(w, 4),
                        "contribution": round(contrib, 4),
                        "direction": direction,
                    })

            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

            reason_text = self._generate_reason_text(contributions[:5], cluster_name)

            cluster_rank = "N/A"
            percentile = 50
            if cross is not None and "score" in cross.columns and "code" in cross.columns:
                scores = cross["score"].sort_values(ascending=False)
                cluster_total = len(scores)
                rank_match = cross[cross["code"] == code]
                if len(rank_match) > 0:
                    score_val = rank_match["score"].values[0]
                    rank_pos = (scores > score_val).sum() + 1
                    cluster_rank = f"{rank_pos}/{cluster_total}"
                    percentile = round((1 - rank_pos / max(cluster_total, 1)) * 100)

            reasons[code] = {
                "cluster": cluster_name,
                "style": cluster_data.get("style", ""),
                "strategy_desc": cluster_data.get("description", ""),
                "summary": reason_text,
                "top_factors": contributions[:3],
                "cluster_rank": cluster_rank,
                "percentile": percentile,
            }

        return reasons

    def _generate_reason_text(self, top_contributions: list, cluster_name: str) -> str:
        """Generate human-readable recommendation reason from factor contributions"""
        if not top_contributions:
            return f"{cluster_name} strategy composite selection"

        group_count = {}
        for c in top_contributions:
            g = c["group"]
            group_count[g] = group_count.get(g, 0) + 1

        dominant_group = max(group_count, key=group_count.get) if group_count else "unknown"
        favorable = [c for c in top_contributions if c["direction"] == "favorable"]

        if len(favorable) >= 2:
            return f"Strong {dominant_group} advantage with high {favorable[0]['name']} and {favorable[1]['name']}"
        elif favorable:
            return f"Leading in {favorable[0]['name']} within {dominant_group} group"
        else:
            return f"{cluster_name} strategy composite selection"

    # ============================================================
    # 6. Output
    # ============================================================

    def print_picks(self):
        """Display recommended picks"""
        portfolio = self.portfolio
        if portfolio is None or len(portfolio) == 0:
            print("No picks to display")
            return

        if "cluster" in portfolio.columns:
            clusters = portfolio["cluster"].unique()
            for cname in clusters:
                subset = portfolio[portfolio["cluster"] == cname]
                style = subset["cluster_style"].iloc[0] if "cluster_style" in subset.columns else ""
                print(f"\n  [{cname}] ({style})")
                print(f"  {'Rank':>4s} {'Code':8s} {'Name':12s} {'Industry':20s} {'Score':>8s} {'Close':>7s} {'Weight':>7s}")
                print(f"  {'-' * 80}")
                for i, (_, row) in enumerate(subset.iterrows()):
                    self._print_stock_row(i, row, wide_ind=True)
        else:
            print(f"\n  {'Rank':>4s} {'Code':8s} {'Name':12s} {'Industry':10s} {'Score':>8s} {'Close':>7s} {'Weight':>7s}")
            print(f"  {'-' * 70}")
            for i, (_, row) in enumerate(portfolio.iterrows()):
                self._print_stock_row(i, row)

        # Industry distribution
        industries = [self.ind_map.get(c, "Unknown") for c in portfolio["code"]]
        ind_dist = Counter(industries)
        print(f"\n  Industry distribution:")
        for ind, count in ind_dist.most_common():
            pct = count / len(portfolio)
            bar = "#" * max(1, int(pct * 30))
            print(f"    {ind:20s} {count} stocks {bar}")

    def _print_stock_row(self, i: int, row, wide_ind: bool = False):
        code = row["code"]
        name = self.name_map.get(code, "")
        ind = self.ind_map.get(code, "Unknown")
        score = row.get("score", row.get("universal_score", 0))
        close = row.get("close", 0)
        weight = row.get("weight", 0)
        ind_width = 20 if wide_ind else 10
        print(f"  {i+1:>3d}. {code:8s} {name:12s} {ind:{ind_width}s} "
              f"score:{score:+.3f}  close:{close:>7.2f}  weight:{weight:.1%}")

    def save(self, filename: str = "live_picks.csv"):
        """Save recommendation results"""
        portfolio = self.portfolio
        out = portfolio[["code"]].copy()
        if "cluster" in portfolio.columns:
            out["cluster"] = portfolio["cluster"]
        out["name"] = out["code"].map(self.name_map)
        out["industry"] = out["code"].map(self.ind_map)
        out["close"] = portfolio.get("close", 0)
        out["score"] = portfolio.get("score", portfolio.get("universal_score", 0))
        out["weight"] = portfolio.get("weight", 0)
        out.to_csv(ROOT / "data" / filename, index=False, encoding="utf-8-sig")
        print(f"\n  Results saved to data/{filename}")


# ============================================================
# 7. Convenience entry points
# ============================================================

def run_live_prediction(top_n: int = 10, neutralize: bool = False):
    """One-click run multi-style parallel live prediction"""
    engine = LiveEngine()

    print("=" * 70)
    print("  Quantitative Multi-Factor Strategy v4.0 - Multi-Style Parallel Prediction")
    print(f"  Strategy clusters: Value-Defense + Growth-Attack + Quality-Balance")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    engine.load_data()

    print("\n[2/5] Computing factors...")
    engine.compute_factors()

    print("\n[3/5] Multi-strategy IC analysis + factor selection...")
    engine.estimate_ic()

    print(f"\n[4/5] Multi-strategy scoring (neutralize={neutralize})...")
    engine.score(neutralize=neutralize)

    print("\n[5/5] Multi-cluster ensemble + portfolio construction + risk control...")
    engine.build()

    engine.print_picks()
    engine.save()

    return engine


if __name__ == "__main__":
    run_live_prediction()
