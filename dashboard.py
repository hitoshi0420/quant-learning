"""
量化数据仪表盘 — 沪深 300 全维度数据浏览器

字段: OHLCV + PE/PB/PS/PCF + 换手率 + 行业分类 + 股票名称

用法: python dashboard.py → 浏览器打开 http://localhost:8050
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from functools import wraps
import threading

import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify, session
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils

app = Flask(__name__)
_secret = os.environ.get("DASHBOARD_SECRET_KEY")
if not _secret:
    _key_file = Path(__file__).resolve().parent / ".secret_key"
    try:
        _secret = _key_file.read_text().strip()
    except FileNotFoundError:
        import secrets
        _secret = secrets.token_urlsafe(32)
        _key_file.write_text(_secret)
app.secret_key = _secret
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 7  # 7 天过期

@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

ROOT = Path(__file__).resolve().parent

# 确保 scripts 目录在路径中
sys.path.insert(0, str(ROOT / "scripts"))

DATA_DIR = ROOT / "data" / "daily"

# ============================================================
# 每日自动同步（后台非阻塞）
# ============================================================
from daily_sync import start_background_sync, get_sync_status, get_stock_sync_time, trigger_manual_sync, get_data_summary
from daily_sync import _run_historical_sync as _run_hist_sync
from auth import init_db, register_user, login_user as auth_login, get_user_by_id, get_watchlist, add_to_watchlist, remove_from_watchlist, admin_list_users, admin_add_user, admin_delete_user, admin_get_user_watchlist
from paper_trading_engine import get_engine, get_rec_status, _load_cached_recommendation

init_db()

print("检查数据是否需要更新...")
sync_result = start_background_sync()
print(f"  {sync_result['message']}")

# ============================================================
# 数据加载
# ============================================================
print("正在加载数据...")

STOCKS = {}
all_data = []

for f in sorted(DATA_DIR.glob("*.parquet")):
    code = f.stem
    df = pd.read_parquet(f)
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["ret"] = df["close"].pct_change()
    STOCKS[code] = df
    all_data.append(df)

FULL = pd.concat(all_data, ignore_index=True)

# 加载名称和行业（从本地 CSV，避免与后台同步争抢 Baostock 连接）
names_df = pd.read_csv(ROOT / "data" / "stock_names.csv", dtype=str)
NAMES = dict(zip(names_df.iloc[:, 0], names_df.iloc[:, 1]))
try:
    ind_df = pd.read_csv(ROOT / "data" / "industry_map.csv", dtype=str)
    INDUSTRIES = dict(zip(ind_df.iloc[:, 0], ind_df.iloc[:, 1]))
except Exception:
    INDUSTRIES = {}

# ============================================================
# 财务数据加载（在SUMMARY之前，为了取ROE）
# ============================================================
print("正在加载财务数据...")
FIN_DIR = ROOT / "data" / "financial"
FINANCIALS = {}
fin_all = []

for f in sorted(FIN_DIR.glob("*.parquet")):
    code = f.stem
    df = pd.read_parquet(f)
    if df.empty:
        continue
    if "stat_date" in df.columns:
        df["stat_date"] = pd.to_datetime(df["stat_date"])
    if "pub_date" in df.columns:
        df["pub_date"] = pd.to_datetime(df["pub_date"])
    FINANCIALS[code] = df
    fin_all.append(df)

FIN_FULL = pd.concat(fin_all, ignore_index=True) if fin_all else pd.DataFrame()

# 构建摘要
SUMMARY = []
today = datetime.now().strftime("%Y-%m-%d")
# 找出全局最新数据日期，用于判断每只股票的延迟程度
all_last_dates = [str(df["date"].max())[:10] for df in STOCKS.values() if len(df) > 0]
if all_last_dates:
    from collections import Counter
    date_counts = Counter(all_last_dates)
    global_latest_date = max(all_last_dates)
else:
    global_latest_date = today

for code, df in STOCKS.items():
    if len(df) < 2:
        continue
    ret_total = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    sharpe = (df["ret"].mean() / df["ret"].std() * np.sqrt(252)) if df["ret"].std() > 0 else 0
    last_date = str(df["date"].max())[:10]
    # ROE 从最新财报提取
    roe = None
    if code in FINANCIALS:
        fin_df = FINANCIALS[code]
        if "roe_avg" in fin_df.columns and not fin_df.empty:
            latest_roe = fin_df.sort_values("stat_date")["roe_avg"].dropna()
            if len(latest_roe) > 0:
                roe = round(float(latest_roe.iloc[-1]), 1)
        elif "roe" in fin_df.columns and not fin_df.empty:
            latest_roe = fin_df.sort_values("stat_date")["roe"].dropna()
            if len(latest_roe) > 0:
                roe = round(float(latest_roe.iloc[-1]), 1)
    SUMMARY.append({
        "code": code,
        "name": NAMES.get(code, ""),
        "industry": INDUSTRIES.get(code, "未知"),
        "rows": len(df),
        "latest": round(df["close"].iloc[-1], 2),
        "return": round(float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 1),
        "pct_change": round(float(df.iloc[-1].get("pct_change", 0) or 0), 2),
        "sharpe": round(sharpe, 2),
        "pe": round(df["pe_ttm"].iloc[-1], 1) if "pe_ttm" in df.columns and not pd.isna(df["pe_ttm"].iloc[-1]) else None,
        "pb": round(df["pb_mrq"].iloc[-1], 1) if "pb_mrq" in df.columns and not pd.isna(df["pb_mrq"].iloc[-1]) else None,
        "turnover": round(df["turnover"].mean(), 2) if "turnover" in df.columns else 0,
        "roe": roe,
        "last_date": last_date,
        "is_fresh": last_date >= global_latest_date,
    })

SUMMARY_DF = pd.DataFrame(SUMMARY).sort_values("code")
CODES = sorted(STOCKS.keys())

print(f"加载完毕: {len(STOCKS)} 只股票, {len(FULL):,} 行, {len(INDUSTRIES)} 个行业, 全局最新: {global_latest_date}")

# 构建财务摘要（补充 SUMMARY 中未包含的财务字段）
for code, df in FINANCIALS.items():
    if code not in STOCKS:
        continue
    latest = df.sort_values("stat_date").iloc[-1] if len(df) > 0 else None
    if latest is not None:
        for s in SUMMARY:
            if s["code"] == code:
                if s.get("roe") is None:
                    s["roe"] = round(float(latest.get("roe_avg", 0) or 0), 1)
                s["np_margin"] = round(float(latest.get("np_margin", 0) or 0), 1)
                s["debt_ratio"] = round(float(latest.get("liability_to_asset", 0) or 0), 1)
                s["yoy_ni"] = round(float(latest.get("yoy_ni", 0) or 0), 1)
                break

print(f"财务加载完毕: {len(FINANCIALS)} 只股票, {len(FIN_FULL):,} 行")

# ============================================================
# HTML
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化数据仪表盘 v2.24.1</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
:root{
  --bg:#0d1117;--surface:#161b22;--hover:#1c2129;--border:#30363d;
  --text:#c9d1d9;--text2:#8b949e;--text3:#484f58;
  --accent:#58a6ff;--green:#3fb950;--red:#f85149;--orange:#f0883e;
  --input-bg:#0d1117;--btn-success:#238636;--btn-danger:#da3633;
  --progress-bg:#21262d;--warning-bg:#1a2332;--shadow:rgba(0,0,0,0.4);
  --wl-added-bg:#1a3a2a;
}
[data-theme="light"]{
  --bg:#ffffff;--surface:#f6f8fa;--hover:#eaedf0;--border:#d0d7de;
  --text:#1f2328;--text2:#656d76;--text3:#8b949e;
  --accent:#0969da;--green:#1a7f37;--red:#cf222e;--orange:#bc4c00;
  --input-bg:#ffffff;--btn-success:#1a7f37;--btn-danger:#cf222e;
  --progress-bg:#e1e4e8;--warning-bg:#fff8c5;--shadow:rgba(0,0,0,0.1);
  --wl-added-bg:#dafbe1;
}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--text3)}
html,body{height:100vh;overflow:hidden}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Microsoft YaHei',sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column}
.header{background:var(--surface);border-bottom:1px solid var(--border);padding:6px 20px;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:16px;color:var(--accent)}
.header .stats{font-size:12px;color:var(--text2)}
.info-btn{background:transparent;border:1px solid var(--border);color:var(--text2);font-size:11px;padding:3px 10px;border-radius:4px;cursor:pointer}
.info-btn:hover{background:var(--hover);border-color:var(--accent);color:var(--accent)}
/* 数据概况弹窗 */
.stats-modal-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;width:360px}
.stats-modal-box h3{font-size:16px;color:var(--accent);margin-bottom:16px}
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.stats-item{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px}
.stats-item .lbl{font-size:10px;color:var(--text2);text-transform:uppercase;margin-bottom:4px}
.stats-item .val{font-size:18px;font-weight:700;color:var(--text)}

/* 标签导航 */
.tabs{display:flex;background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px}
.tab-btn{padding:10px 20px;font-size:13px;color:var(--text2);cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;transition:all 0.2s}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

.tab-content{display:none}
.tab-content.active{display:flex;flex:1;height:100%;overflow:hidden}

/* 仪表盘 */
.dash-layout{display:flex;height:100%;width:100%}
.sidebar{width:340px;background:var(--surface);border-right:1px solid var(--border);overflow-y:auto;display:flex;flex-direction:column;flex-shrink:0;scroll-behavior:smooth;will-change:scroll-position}
.sidebar input{width:100%;padding:10px 14px;background:var(--bg);border:none;border-bottom:1px solid var(--border);color:var(--text);font-size:14px;outline:none}
.sidebar input:focus{border-bottom-color:var(--accent)}
#stock-list{flex:1;overflow-y:auto;padding:4px}
.stock-item{display:flex;justify-content:space-between;align-items:center;padding:9px 12px;border-radius:4px;cursor:pointer;font-size:12px;border-left:3px solid transparent;margin-bottom:1px;transition:background 0.1s,border-color 0.15s}
.stock-item:hover{background:var(--hover)}
.stock-item.active{background:var(--hover);border-left-color:var(--accent)}
.stock-item .info{display:flex;flex-direction:column;gap:2px}
.stock-item .code{font-weight:600;font-size:13px}
.stock-item .name{color:var(--text2);font-size:11px}
.stock-item .right{text-align:right;display:flex;flex-direction:column;gap:2px}
.stock-item .ret{font-weight:600;font-size:12px}
.stock-item .extra{color:var(--text2);font-size:10px}
.sort-btns{display:flex;gap:4px;padding:6px 8px;flex-wrap:wrap}
.sort-btn{font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid var(--border);background:var(--bg);color:var(--text2);cursor:pointer;white-space:nowrap}
.sort-btn:hover{border-color:var(--accent);color:var(--text)}
.sort-btn.active{background:var(--accent);color:white;border-color:var(--accent)}
.market-bar{display:flex;align-items:center;gap:12px;padding:6px 24px;background:var(--surface);border-bottom:1px solid var(--border);font-size:12px;flex-wrap:wrap}
.market-title{color:var(--accent);font-weight:600;margin-right:4px}
.market-item{font-weight:500}
.market-item.up{color:var(--red)}
.market-item.down{color:var(--green)}
.market-item.flat{color:var(--text2)}
.fresh-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle;flex-shrink:0}
.ret.pos{color:var(--red)}.ret.neg{color:var(--green)}
.main{flex:1;overflow-y:auto;padding:20px;scroll-behavior:smooth}
.metrics{display:grid;grid-template-columns:1fr;gap:10px;margin-bottom:16px}

/* 价格头部 */
.price-header{display:flex;align-items:baseline;gap:12px;padding:8px 0}
.ph-name{font-size:18px;font-weight:600;color:var(--text)}
.ph-price{font-size:28px;font-weight:700}
.ph-pct{font-size:16px;font-weight:600}
.ph-date{font-size:12px;color:var(--text2);margin-left:auto}

/* 7列2行指标 */
.detail-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:0;border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:10px}
.dg-item{background:var(--surface);border-right:1px solid var(--border);border-bottom:1px solid var(--border);padding:8px 10px;display:flex;flex-direction:column}
.dg-item:nth-child(7n){border-right:none}
.dg-item:nth-child(n+8){border-bottom:none}
.dg-label{font-size:11px;color:var(--text2);margin-bottom:2px}
.dg-val{font-size:14px;font-weight:500;color:var(--text)}

/* 财务摘要 */
.fin-summary{display:flex;flex-wrap:wrap;gap:8px 16px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:12px;color:var(--text2)}
.fin-summary span{white-space:nowrap}
.val-pos{color:var(--red)}.val-neg{color:var(--green)}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.chart-panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;transition:border-color 0.25s}
.chart-panel:hover{border-color:var(--text3)}
.chart-panel.full{grid-column:1/-1}
.chart-panel h3{font-size:14px;margin-bottom:10px;color:var(--accent)}
#toast{position:fixed;top:16px;right:16px;background:var(--btn-success);color:white;padding:10px 20px;border-radius:6px;display:none;z-index:1000;font-size:13px}
#sync-banner{background:var(--warning-bg);border-bottom:1px solid var(--border);padding:6px 24px;font-size:12px;color:var(--text2);display:flex;align-items:center;gap:8px}
#sync-banner .dot{width:8px;height:8px;border-radius:50%}
#sync-banner .dot.ok{background:var(--green)}
#sync-banner .dot.syncing{background:var(--orange);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}

/* 同步页面 */
.sync-page{padding:30px 40px;overflow-y:auto;height:calc(100vh - 110px);width:100%}
.sync-page h2{font-size:20px;color:var(--accent);margin-bottom:24px}
.phase-indicator{display:flex;gap:4px;margin-bottom:30px;flex-wrap:wrap;align-items:center}
.phase-step{display:flex;align-items:center;gap:6px;padding:8px 16px;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:12px;color:var(--text3)}
.phase-step.done{color:var(--green);border-color:var(--btn-success)}
.phase-step.active{color:var(--accent);border-color:var(--accent);background:var(--warning-bg)}
.phase-arrow{color:var(--border);font-size:12px}
.progress-section{margin-bottom:28px}
.progress-section h3{font-size:14px;color:var(--text);margin-bottom:12px}
.progress-bar-bg{background:var(--progress-bg);border-radius:4px;height:20px;overflow:hidden;margin-bottom:8px}
.progress-bar-fill{background:var(--btn-success);height:100%;border-radius:4px;transition:width 0.8s ease}
.progress-bar-fill.running{background:var(--accent)}
.progress-stats{display:flex;justify-content:space-between;font-size:12px;color:var(--text2)}
.sync-log{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-top:20px;max-height:200px;overflow-y:auto;font-size:11px;color:var(--text2);font-family:monospace;line-height:1.6}
.sync-log .log-ok{color:var(--green)}
.sync-log .log-warn{color:var(--orange)}

/* 策略占位页 */
.placeholder-page{display:flex;align-items:center;justify-content:center;height:calc(100vh - 110px);width:100%;flex-direction:column}
.placeholder-page .icon{font-size:48px;margin-bottom:16px;opacity:0.3}
.placeholder-page h2{font-size:18px;color:var(--text2);margin-bottom:8px}
.placeholder-page p{font-size:13px;color:var(--text3)}

/* 登录/注册 */
#auth-area{display:flex;align-items:center;gap:8px}
.btn-sm{background:var(--btn-success);color:white;border:none;padding:5px 14px;border-radius:5px;font-size:12px;cursor:pointer}
.btn-sm:hover{opacity:0.85}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn-outline:hover{background:var(--hover)}
.btn-danger{background:var(--btn-danger);color:white;border:none;padding:3px 10px;border-radius:4px;font-size:11px;cursor:pointer}
.btn-danger:hover{opacity:0.85}
.user-badge{font-size:12px;color:var(--accent);font-weight:600}

/* 模态框 */
.modal-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);z-index:2000;display:flex;align-items:center;justify-content:center}
.modal-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;width:340px}
.modal-box h3{font-size:18px;color:var(--text);margin-bottom:16px}
.modal-box input{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-size:14px;margin-bottom:10px;box-sizing:border-box}
.modal-box input:focus{outline:none;border-color:var(--accent)}
.modal-btns{display:flex;gap:8px;margin-top:4px}

.wl-add-btn{background:transparent;border:1px solid var(--border);color:var(--text2);font-size:10px;padding:2px 6px;border-radius:4px;cursor:pointer;margin-left:6px;flex-shrink:0}
.wl-add-btn:hover{background:var(--hover);border-color:var(--accent);color:var(--accent)}
.wl-add-btn.added{background:var(--wl-added-bg);border-color:var(--btn-success);color:var(--green)}

/* 自选股布局 */
.wl-layout{display:flex;height:calc(100vh - 110px);width:100%}
.wl-sidebar{width:320px;border-right:1px solid var(--border);padding:20px;overflow-y:auto;flex-shrink:0}
.wl-sidebar h3{font-size:15px;color:var(--accent);margin-bottom:12px}
.wl-main{flex:1;padding:20px 30px;overflow-y:auto}
.wl-main h3{font-size:15px;color:var(--accent);margin-bottom:12px}
#wl-search{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-size:13px;margin-bottom:10px;box-sizing:border-box}
#wl-search:focus{outline:none;border-color:var(--accent)}
.wl-results{font-size:12px}
.wl-item{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;border:1px solid var(--border);border-radius:6px;margin-bottom:6px;background:var(--bg)}
.wl-item .code{color:var(--accent);font-weight:600;margin-right:8px}
.wl-item .name{color:var(--text);flex:1}
.wl-item .extra{color:var(--text2);margin-right:8px;font-size:11px}
.wl-item .btn-sm{padding:3px 10px;font-size:11px}

/* 管理员页面 */
.admin-page{padding:30px 40px;overflow-y:auto;height:calc(100vh - 110px);width:100%}
.admin-page h2{font-size:20px;color:var(--orange);margin-bottom:20px}
.admin-toolbar{display:flex;align-items:center;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.admin-toolbar input{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:6px 10px;color:var(--text);font-size:13px;width:120px}
.admin-toolbar input:focus{outline:none;border-color:var(--orange)}
#admin-users-table{font-size:13px}
.admin-row{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border:1px solid var(--border);border-radius:6px;margin-bottom:6px;background:var(--bg)}
.admin-row .uname{color:var(--text);font-weight:600;min-width:120px}
.admin-row .meta{color:var(--text2);font-size:12px;flex:1}
.admin-wl-detail{margin-top:12px;font-size:12px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px}

/* 同步页面增强 */
.sync-summary{display:flex;gap:16px;margin-bottom:24px}
.ss-item{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;display:flex;flex-direction:column;gap:4px}
.ss-label{font-size:11px;color:var(--text2);text-transform:uppercase}
.ss-val{font-size:18px;color:var(--accent);font-weight:600}
.ss-sub{font-size:11px;color:var(--text3)}
.sync-actions{display:flex;align-items:center;gap:12px;margin-bottom:24px;padding:16px;background:var(--surface);border:1px solid var(--border);border-radius:8px}
.sync-actions .btn-sm{padding:10px 24px;font-size:14px}
.sync-historical{padding:16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;margin-bottom:24px}

	/* 策略交易 */
	.strategy-layout{display:flex;flex-direction:column;gap:12px;padding:16px 20px;height:100%;overflow:hidden}
	.strategy-row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;flex:1;min-height:0}
	.strategy-col{min-width:0;overflow:hidden;display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
	.chart-modal-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;width:700px;max-width:calc(100vw - 32px);max-height:90vh;overflow-y:auto}
	/* 推荐卡片 */
	.pick-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;margin-bottom:8px;overflow:hidden;font-size:13px;transition:border-color 0.2s}
	.pick-card:hover{border-color:var(--accent)}
	.pick-card-header{display:flex;align-items:center;gap:8px;padding:10px 12px;background:var(--surface);border-bottom:1px solid var(--border)}
	.pick-rank{font-size:12px;font-weight:700;color:var(--text2);min-width:24px}
	.pick-code-name{flex:1;font-size:13px}
	.pick-style-badge{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:500}
	.pick-percentile{font-size:11px;font-weight:600;white-space:nowrap}
	.pick-card-body{padding:10px 12px}
	.pick-meta-row{display:flex;flex-wrap:wrap;gap:4px 12px;margin-bottom:6px}
	.pick-meta-item{font-size:11px;color:var(--text2)}
	.pick-reason{font-size:12px;color:var(--orange);margin-bottom:6px;font-style:italic;padding:4px 0}
	.factor-tags{display:flex;flex-wrap:wrap;gap:4px}
	.factor-tag{font-size:10px;padding:2px 6px;border-radius:4px;white-space:nowrap}
	.factor-tag-good{background:#00c85320;color:#00c853;border:1px solid #00c85340}
	.factor-tag-bad{background:#ff525220;color:#ff5252;border:1px solid #ff525240}
	.pick-card-footer{padding:6px 12px;border-top:1px solid var(--border);display:flex;gap:6px}
	.btn-xs{font-size:10px;padding:3px 8px;border-radius:3px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer}
	.btn-xs:hover{background:var(--accent);color:white}
	.pos-row{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;border:1px solid var(--border);border-radius:6px;margin-bottom:4px;font-size:12px;background:var(--bg)}
	.pos-code{color:var(--accent);font-weight:600;min-width:70px}
	.pos-shares{color:var(--text);min-width:60px;text-align:right}
	.pos-value{color:var(--text);min-width:80px;text-align:right}
	.txn-row{display:flex;align-items:center;gap:10px;padding:6px 10px;border-bottom:1px solid var(--border);font-size:12px}
	.txn-row:last-child{border-bottom:none}
	/* 买入弹窗 */
	.buy-modal-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;width:400px;max-width:calc(100vw - 32px);max-height:80vh;overflow-y:auto}
	.buy-modal-box h3{font-size:18px;color:var(--text);margin-bottom:16px}
	.buy-stock-info{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:16px}
	.buy-stock-info .info-row{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;color:var(--text2)}
	.buy-stock-info .info-row strong{color:var(--text)}
	.buy-calc{font-size:13px;color:var(--text);padding:10px;background:var(--bg);border-radius:6px;margin:10px 0;border:1px solid var(--border)}
	/* 一键买入摘要 */
	.basket-summary{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin:12px 0;max-height:250px;overflow-y:auto}
	.basket-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px}
	.basket-row:last-child{border-bottom:none}
	.pos-clickable{cursor:pointer;transition:border-color 0.2s}
	.pos-clickable:hover{border-color:var(--accent)!important}
	/* 股票详情弹窗 */
	.detail-modal-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;width:480px;max-width:calc(100vw - 32px);max-height:85vh;overflow-y:auto}
	.detail-modal-box h3{font-size:18px;color:var(--accent);margin-bottom:8px}
	.detail-section{margin-bottom:16px}
	.detail-section h4{font-size:13px;color:var(--accent);margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--border)}
	.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px 16px}
	.detail-item{display:flex;justify-content:space-between;font-size:12px;padding:3px 0}
	.detail-item .lbl{color:var(--text2)}
</style>
</head>
<body>
<div class="header">
  <h1>A股量化数据平台 <span style="font-size:11px;color:var(--text3);font-weight:400">v2.24.1</span></h1>
  <button class="info-btn" onclick="showStatsModal()" title="数据概况">ℹ 数据概况</button>
  <div id="auth-area">
    <button class="btn-sm btn-outline" id="btn-theme" onclick="toggleTheme()" title="切换主题" style="margin-right:8px">☀</button>
    <span id="auth-guest">
      <button class="btn-sm" onclick="showLoginModal()">登录</button>
      <button class="btn-sm btn-outline" onclick="showRegisterModal()">注册</button>
    </span>
    <span id="auth-logged" style="display:none">
      <span class="user-badge" id="auth-username"></span>
      <button class="btn-sm btn-outline" onclick="handleLogout()">登出</button>
    </span>
  </div>
</div>

<!-- 登录/注册模态框 -->
<div class="modal-overlay" id="login-modal" style="display:none">
  <div class="modal-box">
    <h3 id="modal-title">登录</h3>
    <input type="text" id="modal-username" placeholder="用户名" autocomplete="username">
    <input type="password" id="modal-password" placeholder="密码" autocomplete="current-password">
    <div id="modal-error" style="color:var(--red);font-size:12px;min-height:18px"></div>
    <div class="modal-btns">
      <button class="btn-sm" id="modal-submit-btn" onclick="handleAuthSubmit()">登录</button>
      <button class="btn-sm btn-outline" onclick="hideLoginModal()">取消</button>
    </div>
    <div style="font-size:12px;color:var(--text2);margin-top:8px;text-align:center">
      <span id="modal-switch-text">没有账号？</span>
      <a href="#" onclick="toggleAuthMode();return false" style="color:var(--accent)" id="modal-switch-link">去注册</a>
    </div>
  </div>
</div>
<div id="sync-banner" style="display:none"><span class="dot ok" id="sync-dot"></span><span id="sync-text"></span></div>
<div class="market-bar" id="market-bar">
  <span class="market-title">市场概况</span>
  <span class="market-item" id="mkt-avg-pct">加载中...</span>
  <span class="market-item up" id="mkt-up">--</span>
  <span class="market-item down" id="mkt-down">--</span>
  <span class="market-item flat" id="mkt-flat">--</span>
  <span style="color:var(--text3);margin:0 4px">|</span>
  <span class="market-item" id="mkt-amount" style="font-size:11px">--</span>
  <span class="market-item" id="mkt-pe" style="font-size:11px">--</span>
  <span class="market-item" id="mkt-turnover" style="font-size:11px">--</span>
  <span style="color:var(--text3);margin:0 4px">|</span>
  <span class="market-item" id="mkt-hot-sectors" style="font-size:11px;color:var(--red)">--</span>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('dash')">仪表盘</button>
  <button class="tab-btn admin-only" id="tab-btn-sync" onclick="switchTab('sync')" style="display:none">数据同步</button>
  <button class="tab-btn" onclick="switchTab('strategy')">量化策略选股</button>
  <button class="tab-btn" onclick="switchTab('watchlist')">自选股</button>
  <button class="tab-btn admin-only" id="tab-btn-admin" onclick="switchTab('admin')" style="display:none">管理员</button>
</div>

<!-- 数据概况弹窗 -->
<div class="modal-overlay" id="stats-modal" style="display:none">
  <div class="stats-modal-box">
    <h3>数据概况</h3>
    <div class="stats-grid">
      <div class="stats-item"><div class="lbl">股票数量</div><div class="val" id="st-n-stocks">-</div></div>
      <div class="stats-item"><div class="lbl">数据行数</div><div class="val" id="st-n-rows">-</div></div>
      <div class="stats-item"><div class="lbl">行业数量</div><div class="val" id="st-n-ind">-</div></div>
      <div class="stats-item"><div class="lbl">数据至</div><div class="val" id="st-latest" style="font-size:13px">-</div></div>
      <div class="stats-item"><div class="lbl">同步状态</div><div class="val" id="st-sync" style="font-size:13px;color:var(--accent)">-</div></div>
      <div class="stats-item"><div class="lbl">同步详情</div><div class="val" id="st-sync-detail" style="font-size:10px;color:var(--text2)">-</div></div>
    </div>
    <div style="margin-top:16px;text-align:right">
      <button class="btn-sm btn-outline" onclick="hideStatsModal()">关闭</button>
    </div>
  </div>
</div>

<!-- 标签1: 仪表盘 -->
<div class="tab-content active" id="tab-dash">
  <div class="dash-layout">
    <div class="sidebar">
      <input type="text" id="search" placeholder="搜索代码或名称..." oninput="filterStocks()">
      <div class="sort-btns">
        <button class="sort-btn active" data-sort="code" data-label="代码" onclick="setSort('code')">代码↑</button>
        <button class="sort-btn" data-sort="pct" data-label="涨幅" onclick="setSort('pct')">涨幅</button>
        <button class="sort-btn" data-sort="name" data-label="名称" onclick="setSort('name')">名称</button>
      </div>
      <div id="stock-list"></div>
    </div>
    <div class="main" id="main-content">
      <div class="metrics" id="metrics-price"></div>
      <div class="metrics" id="metrics-fin" style="margin-bottom:16px"></div>
      <div class="chart-row">
        <div class="chart-panel"><h3>走势 & 估值</h3><div id="chart-price"></div></div>
        <div class="chart-panel"><h3>日收益率分布</h3><div id="chart-ret"></div></div>
      </div>
      <div class="chart-row">
        <div class="chart-panel"><h3>成交量</h3><div id="chart-vol"></div></div>
        <div class="chart-panel"><h3>换手率 & PE</h3><div id="chart-val"></div></div>
      </div>
      <div class="chart-row">
        <div class="chart-panel"><h3>ROE & 净利率趋势</h3><div id="chart-roe"></div></div>
        <div class="chart-panel"><h3>杜邦分析</h3><div id="chart-dupont"></div></div>
      </div>
    </div>
  </div>
</div>

<!-- 标签2: 数据同步 -->
<div class="tab-content" id="tab-sync">
  <div class="sync-page">
    <h2>数据同步状态</h2>

    <!-- 数据概览 -->
    <div class="sync-summary" id="sync-summary">
      <div class="ss-item"><span class="ss-label">日线数据</span><span class="ss-val" id="ss-daily-range">加载中...</span><span class="ss-sub" id="ss-daily-count"></span></div>
      <div class="ss-item"><span class="ss-label">财务数据</span><span class="ss-val" id="ss-fin-range">加载中...</span><span class="ss-sub" id="ss-fin-count"></span></div>
    </div>

    <!-- 操作按钮 -->
    <div class="sync-actions">
      <button class="btn-sm" id="btn-manual-sync" onclick="triggerManualSync()">手动拉取最新</button>
      <span style="font-size:12px;color:var(--text2);margin-left:8px" id="sync-action-msg"></span>
    </div>

    <!-- 历史拉取 -->
    <div class="sync-historical">
      <h3 style="font-size:13px;color:var(--text2);margin-bottom:8px">拉取历史数据</h3>
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
        <input type="date" id="hist-start" style="background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:4px 8px;color:var(--text);font-size:12px">
        <span style="color:var(--text2);font-size:12px">至</span>
        <input type="date" id="hist-end" style="background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:4px 8px;color:var(--text);font-size:12px">
        <button class="btn-sm btn-outline" onclick="triggerHistoricalSync()">拉取</button>
        <span style="font-size:12px;color:var(--text2)" id="hist-msg"></span>
      </div>
    </div>

    <!-- 进度指示 -->
    <div class="phase-indicator" id="phase-steps">
      <div class="phase-step" id="phase-names">股票名称</div>
      <span class="phase-arrow">&rarr;</span>
      <div class="phase-step" id="phase-industries">行业分类</div>
      <span class="phase-arrow">&rarr;</span>
      <div class="phase-step" id="phase-daily">日线数据</div>
      <span class="phase-arrow">&rarr;</span>
      <div class="phase-step" id="phase-financial">财务数据</div>
      <span class="phase-arrow">&rarr;</span>
      <div class="phase-step" id="phase-done">完成</div>
    </div>
    <div style="margin-bottom:20px;padding:12px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px">
      <span style="color:var(--text2);font-size:11px">实时状态</span><br>
      <span id="cur-stock" style="font-size:15px;font-weight:600;color:var(--accent)">等待中...</span>
      <span id="cur-type" style="font-size:12px;color:var(--text2);margin-left:12px"></span>
    </div>
    <div class="progress-section">
      <h3>日线数据</h3>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="daily-bar" style="width:0%"></div></div>
      <div class="progress-stats"><span id="daily-text">等待中...</span><span id="daily-pct">0%</span></div>
    </div>
    <div class="progress-section">
      <h3>财务数据</h3>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="fin-bar" style="width:0%"></div></div>
      <div class="progress-stats"><span id="fin-text">等待中...</span><span id="fin-pct">0%</span></div>
    </div>
    <div class="sync-log" id="sync-log">
      <div>> 等待同步启动...</div>
    </div>
  </div>
</div>

  <!-- 标签3: 量化策略选股 + 纸交易 -->
  <div class="tab-content" id="tab-strategy">
    <div id="strategy-guest-view" style="display:flex;height:calc(100vh - 110px);width:100%">
      <div class="placeholder-page">
        <div class="icon">&#9883;</div>
        <h2>量化策略选股</h2>
        <p>登录后可使用多因子量化策略、模拟交易等功能</p>
        <button class="btn-sm" onclick="showLoginModal()" style="margin-top:12px">立即登录</button>
      </div>
    </div>
    <div id="strategy-user-view" style="display:none">
    <div class="strategy-layout">
      <!-- 第一行: 策略推荐 + 虚拟账户 + 交易记录 三列并排 -->
      <div class="strategy-row3">
        <!-- 列1: 策略推荐 -->
        <div class="strategy-col" style="padding:14px 16px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <h3 style="font-size:14px;color:var(--accent)">策略推荐 (v4.0 多因子)</h3>
            <div style="display:flex;gap:6px">
              <button class="btn-sm" id="btn-basket-buy" onclick="showBasketModal()" style="background:var(--red);color:white;display:none;font-size:11px">一键买入</button>
              <button class="btn-sm" id="btn-refresh-picks" onclick="refreshPicks()" style="font-size:11px">刷新推荐</button>
            </div>
          </div>
          <div id="rec-progress-wrap" style="display:none;margin-bottom:10px;padding:8px 10px;background:var(--bg);border:1px solid var(--border);border-radius:6px">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
              <span id="rec-progress-step" style="font-size:10px;color:var(--accent)">初始化...</span>
              <span id="rec-progress-pct" style="font-size:10px;color:var(--text2)">0%</span>
            </div>
            <div class="progress-bar-bg" style="height:3px;margin-bottom:2px"><div class="progress-bar-fill" id="rec-progress-bar" style="width:0%;height:3px;background:var(--accent)"></div></div>
            <div id="rec-progress-msg" style="font-size:10px;color:var(--text2)"></div>
          </div>
          <div id="strategy-picks" style="overflow-y:auto;flex:1;font-size:12px">点击"刷新推荐"获取今日策略信号...</div>
        </div>

        <!-- 列2: 虚拟账户 -->
        <div class="strategy-col">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <h3 style="font-size:14px;color:var(--accent);margin:0">我的虚拟账户</h3>
            <div style="display:flex;gap:6px">
              <button class="btn-sm btn-outline" onclick="showPnlChartModal()" style="font-size:11px">走势图</button>
              <button class="btn-sm" onclick="showRechargeModal()" style="font-size:11px">充值</button>
              <button class="btn-sm btn-outline" onclick="resetAccount()" style="color:var(--red);border-color:var(--red);font-size:11px">重置</button>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;margin-bottom:12px">
            <div><span style="font-size:10px;color:var(--text2)">总资产</span><br><span style="font-size:18px;font-weight:700;color:var(--accent)" id="acct-total">¥0.00</span></div>
            <div><span style="font-size:10px;color:var(--text2)">可用现金</span><br><span style="font-size:15px;font-weight:600" id="acct-cash">¥0.00</span></div>
            <div><span style="font-size:10px;color:var(--text2)">持仓市值</span><br><span style="font-size:15px;font-weight:600" id="acct-mktval">¥0.00</span></div>
            <div><span style="font-size:10px;color:var(--text2)">累计盈亏</span><br><span style="font-size:15px;font-weight:600" id="acct-pnl">¥0.00</span></div>
          </div>
          <div style="font-size:11px;border-top:1px solid var(--border);padding-top:8px">
            <span style="color:var(--text2)">今日盈亏 </span>
            <span style="font-weight:600" id="acct-today-pnl">¥0.00</span>
          </div>

          <!-- 当前持仓 -->
          <div style="margin-top:14px;border-top:1px solid var(--border);padding-top:10px;flex:1;display:flex;flex-direction:column;min-height:0">
            <h4 style="font-size:13px;color:var(--accent);margin-bottom:6px;flex-shrink:0">当前持仓</h4>
            <div id="positions-list" style="font-size:11px;color:var(--text2);overflow-y:auto;flex:1">暂无持仓</div>
          </div>
        </div>

        <!-- 列3: 交易记录 -->
        <div class="strategy-col">
          <h3 style="font-size:14px;color:var(--accent);margin-bottom:10px;flex-shrink:0">交易记录</h3>
          <div id="txn-list" style="font-size:11px;overflow-y:auto;flex:1">加载中...</div>
        </div>
      </div>

    </div>
    </div><!-- strategy-user-view -->
  </div>

<!-- 走势图模态框 -->
<div class="modal-overlay" id="pnl-chart-modal" style="display:none">
  <div class="chart-modal-box">
    <h3>盈亏走势</h3>
    <div id="chart-pnl-amt" style="height:240px;margin-bottom:12px"></div>
    <h3 style="margin-top:8px">收益率走势</h3>
    <div id="chart-pnl-pct" style="height:240px"></div>
    <div style="margin-top:16px;text-align:right">
      <button class="btn-sm btn-outline" onclick="hidePnlChartModal()">关闭</button>
    </div>
  </div>
</div>

<!-- 充值模态框 -->
<div class="modal-overlay" id="recharge-modal" style="display:none">
  <div class="modal-box">
    <h3>充值虚拟币</h3>
    <p style="font-size:12px;color:var(--text2);margin-bottom:12px">虚拟资金，仅供模拟交易使用</p>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px">
      <button class="btn-sm btn-outline" onclick="quickRecharge(10000)">¥10,000</button>
      <button class="btn-sm btn-outline" onclick="quickRecharge(50000)">¥50,000</button>
      <button class="btn-sm btn-outline" onclick="quickRecharge(100000)">¥100,000</button>
      <button class="btn-sm btn-outline" onclick="quickRecharge(500000)">¥500,000</button>
    </div>
    <input type="number" id="recharge-amount" placeholder="自定义金额" min="1" style="width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-size:14px;margin-bottom:10px">
    <div class="modal-btns">
      <button class="btn-sm" onclick="doRecharge()">确认充值</button>
      <button class="btn-sm btn-outline" onclick="hideRechargeModal()">取消</button>
    </div>
    <div id="recharge-msg" style="font-size:12px;color:var(--green);margin-top:8px;min-height:18px"></div>
  </div>
</div>

<!-- 买入弹窗 -->
<div class="modal-overlay" id="buy-modal" style="display:none">
  <div class="buy-modal-box">
    <h3>买入股票</h3>
    <div class="buy-stock-info">
      <div class="info-row"><span>股票代码</span><strong id="bm-code">-</strong></div>
      <div class="info-row"><span>股票名称</span><strong id="bm-name">-</strong></div>
      <div class="info-row"><span>当前价格</span><strong id="bm-price" style="color:var(--red)">-</strong></div>
      <div class="info-row"><span>可用资金</span><strong id="bm-cash" style="color:var(--accent)">-</strong></div>
    </div>
    <div style="margin-bottom:12px">
      <label style="font-size:12px;color:var(--text2)">买入股数（100的倍数）</label>
      <div style="display:flex;gap:8px;margin-top:4px">
        <input type="number" id="bm-shares" min="100" step="100" value="100" style="flex:1;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-size:14px" oninput="updateBuyCalc()">
        <button class="btn-sm btn-outline" onclick="setBuyShares(100)">1手</button>
        <button class="btn-sm btn-outline" onclick="setBuyShares(500)">5手</button>
        <button class="btn-sm btn-outline" onclick="setBuyMaxShares()">全仓</button>
      </div>
    </div>
    <div class="buy-calc" id="bm-calc">预计花费: ¥0.00 | 佣金: ¥5.00</div>
    <div class="modal-btns">
      <button class="btn-sm" onclick="confirmBuy()" style="background:var(--red);color:white;flex:1">确认买入</button>
      <button class="btn-sm btn-outline" onclick="hideBuyModal()">取消</button>
    </div>
    <div id="bm-msg" style="font-size:12px;color:var(--green);margin-top:8px;min-height:18px"></div>
  </div>
</div>

<!-- 一键买入弹窗 -->
<div class="modal-overlay" id="basket-modal" style="display:none">
  <div class="buy-modal-box">
    <h3>一键买入推荐组合</h3>
    <div class="buy-stock-info">
      <div class="info-row"><span>可用资金</span><strong id="bk-cash" style="color:var(--accent)">-</strong></div>
      <div class="info-row"><span>推荐日期</span><strong id="bk-date">-</strong></div>
      <div class="info-row"><span>股票数量</span><strong id="bk-count">-</strong></div>
    </div>
    <div class="basket-summary" id="bk-list">加载中...</div>
    <div class="modal-btns">
      <button class="btn-sm" onclick="confirmBasketBuy()" style="background:var(--red);color:white;flex:1">确认一键买入</button>
      <button class="btn-sm btn-outline" onclick="hideBasketModal()">取消</button>
    </div>
    <div id="bk-msg" style="font-size:12px;color:var(--green);margin-top:8px;min-height:18px"></div>
  </div>
</div>

<!-- 股票详情弹窗 -->
<div class="modal-overlay" id="detail-modal" style="display:none">
  <div class="detail-modal-box">
    <h3 id="dt-name">-</h3>
    <div style="font-size:12px;color:var(--text2);margin-bottom:16px" id="dt-code-industry">-</div>
    <div class="detail-section">
      <h4>行情数据</h4>
      <div class="detail-grid" id="dt-market">加载中...</div>
    </div>
    <div class="detail-section">
      <h4>财务指标</h4>
      <div class="detail-grid" id="dt-fin">加载中...</div>
    </div>
    <div style="text-align:right;margin-top:12px">
      <button class="btn-sm btn-outline" onclick="hideDetailModal()">关闭</button>
    </div>
  </div>
</div>

<!-- 标签4: 自选股 -->
<div class="tab-content" id="tab-watchlist">
  <div class="wl-layout" id="wl-guest-view">
    <div class="placeholder-page">
      <div class="icon">&#9733;</div>
      <h2>我的自选股</h2>
      <p>登录后可管理自选股，跟踪心仪标的</p>
      <button class="btn-sm" onclick="showLoginModal()" style="margin-top:12px">立即登录</button>
    </div>
  </div>
  <div class="wl-layout" id="wl-user-view" style="display:none">
    <!-- 左侧: 搜索 + 自选股列表 -->
    <div class="wl-sidebar">
      <h3>添加自选股</h3>
      <input type="text" id="wl-search" placeholder="输入股票代码或名称..." oninput="searchWatchStocks()">
      <div id="wl-search-results" class="wl-results"></div>
      <h3 style="margin-top:16px">我的自选股 (<span id="wl-count">0</span>)</h3>
      <div class="sort-btns" style="margin:0 0 8px">
        <button class="sort-btn active" data-sort="code" data-label="代码" onclick="setSort('code')">代码↑</button>
        <button class="sort-btn" data-sort="pct" data-label="涨幅" onclick="setSort('pct')">涨幅</button>
        <button class="sort-btn" data-sort="name" data-label="名称" onclick="setSort('name')">名称</button>
      </div>
      <div id="wl-list">加载中...</div>
    </div>
    <!-- 右侧: 股票详情 -->
    <div class="wl-main" id="wl-detail">
      <div class="placeholder-page" id="wl-detail-empty">
        <div class="icon">&#9733;</div>
        <p style="color:var(--text2)">点击左侧自选股查看详情</p>
      </div>
      <div id="wl-detail-content" style="display:none">
        <div class="metrics" id="wl-metrics-price"></div>
        <div class="metrics" id="wl-metrics-fin"></div>
        <div class="chart-row">
          <div class="chart-panel"><h3>走势</h3><div id="wl-chart-price" style="height:300px"></div></div>
          <div class="chart-panel"><h3>收益率分布</h3><div id="wl-chart-ret" style="height:300px"></div></div>
        </div>
        <div class="chart-row">
          <div class="chart-panel"><h3>成交量</h3><div id="wl-chart-vol" style="height:220px"></div></div>
          <div class="chart-panel"><h3>换手率 & PE</h3><div id="wl-chart-val" style="height:220px"></div></div>
        </div>
        <div class="chart-row">
          <div class="chart-panel"><h3>ROE & 净利率趋势</h3><div id="wl-chart-roe" style="height:250px"></div></div>
          <div class="chart-panel"><h3>杜邦分析</h3><div id="wl-chart-dupont" style="height:250px"></div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- 标签5: 管理员 -->
<div class="tab-content" id="tab-admin">
  <div class="admin-page">
    <h2>管理员面板</h2>
    <div class="admin-toolbar">
      <input type="text" id="admin-new-user" placeholder="用户名">
      <input type="password" id="admin-new-pass" placeholder="密码">
      <label style="font-size:12px;color:var(--text2);display:flex;align-items:center;gap:4px"><input type="checkbox" id="admin-new-admin"> 管理员权限</label>
      <button class="btn-sm" onclick="adminAddUser()">添加用户</button>
    </div>
    <div id="admin-users-table"></div>
    <div id="admin-user-wl" class="admin-wl-detail"></div>
  </div>
</div>

<div id="toast"></div>
<script id="stock-list-data" type="application/json">{{ stock_list_json | safe }}</script>
<script>
// HTML 转义，防止 XSS
const escHtml = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
let allStocks = JSON.parse(document.getElementById('stock-list-data').textContent);
let currentCode = '600519';
let compareCode = null;
let activeTab = 'dash';
let syncPollTimer = null;
let _loggedIn = false;
let wlCodes = new Set();
let currentTheme = 'dark';
let plotlyTemplate = 'plotly_dark';
let plotlyPaperBg = '#161b22';
let plotlyPlotBg = '#161b22';
let plotlyGridColor = '#30363d';

function initTheme() {
  const saved = localStorage.getItem('ql-theme');
  if (saved === 'light') { currentTheme = 'light'; }
  applyTheme();
}

function toggleTheme() {
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('ql-theme', currentTheme);
  applyTheme();
  // 用 DOM 实际状态判断当前标签（避免 activeTab 变量不同步）
  const dashActive = document.getElementById('tab-dash').classList.contains('active');
  const wlActive = document.getElementById('tab-watchlist').classList.contains('active');
  if (dashActive && currentCode) selectStock(currentCode);
  if (wlActive) {
    const wlDetail = document.getElementById('wl-detail-content');
    if (wlDetail && wlDetail.style.display !== 'none' && currentCode) viewWatchStock(currentCode);
  }
}

function applyTheme() {
  const isLight = currentTheme === 'light';
  document.documentElement.dataset.theme = isLight ? 'light' : '';
  plotlyTemplate = isLight ? 'plotly_white' : 'plotly_dark';
  plotlyPaperBg = isLight ? '#ffffff' : '#161b22';
  plotlyPlotBg = isLight ? '#f6f8fa' : '#161b22';
  plotlyGridColor = isLight ? '#d0d7de' : '#30363d';
  document.getElementById('btn-theme').textContent = isLight ? '☾' : '☀';
}

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  var tabBtns = document.querySelectorAll('.tab-btn');
  for (var bi = 0; bi < tabBtns.length; bi++) {
    var oc = tabBtns[bi].getAttribute('onclick') || '';
    if (oc.indexOf("'" + tab + "'") !== -1) {
      tabBtns[bi].classList.add('active');
      break;
    }
  }
  if(tab==='sync') {
    pollSyncDetail();
    loadDataSummary();
    syncPollTimer = setInterval(pollSyncDetail, 3000);
  } else {
    if(syncPollTimer) { clearInterval(syncPollTimer); syncPollTimer = null; }
    if(tab==='dash' && currentCode) selectStock(currentCode);
    if(tab==='watchlist') { checkAuthState(); loadWatchlist(); }
    if(tab==='strategy') { checkAuthState().then(() => loadStrategyTab()); }
    if(tab==='admin') loadAdminUsers();
  }
}

// === 仪表盘函数 ===
async function refreshMarketBar() {
  // 快速本地计算（立即显示）
  const up = allStocks.filter(s => (s.pct_change || 0) > 0).length;
  const down = allStocks.filter(s => (s.pct_change || 0) < 0).length;
  const flat = allStocks.length - up - down;
  const avgPct = allStocks.length > 0 ? allStocks.reduce((a, s) => a + (s.pct_change || 0), 0) / allStocks.length : 0;
  document.getElementById('mkt-avg-pct').textContent = '均价 ' + (avgPct >= 0 ? '+' : '') + avgPct.toFixed(2) + '%';
  document.getElementById('mkt-avg-pct').style.color = avgPct >= 0 ? 'var(--red)' : 'var(--green)';
  document.getElementById('mkt-up').textContent = '↑' + up + '家';
  document.getElementById('mkt-down').textContent = '↓' + down + '家';
  document.getElementById('mkt-flat').textContent = '—' + flat + '家';
  // 异步获取详细数据
  try {
    const r = await fetch('/api/market/summary');
    const m = await r.json();
    // 总成交额（亿）
    if (m.total_amount) {
      document.getElementById('mkt-amount').textContent = '成交 ' + (m.total_amount / 1e8).toFixed(0) + '亿';
    }
    if (m.avg_pe) {
      document.getElementById('mkt-pe').textContent = '均PE ' + m.avg_pe;
    }
    document.getElementById('mkt-turnover').textContent = '均换手 ' + m.avg_turnover + '%';
    // 热门板块
    if (m.top_sectors && m.top_sectors.length >= 2) {
      const hot = m.top_sectors.slice(0, 2).map(s => s.name + ' +' + s.pct + '%').join(' ');
      const cold = m.bottom_sectors ? m.bottom_sectors.slice(0, 1).map(s => s.name + ' ' + s.pct + '%').join(' ') : '';
      document.getElementById('mkt-hot-sectors').innerHTML =
        '<span style="color:var(--red)">热点: ' + hot + '</span>'
        + (cold ? ' <span style="color:var(--green)">| 弱势: ' + cold + '</span>' : '');
    }
  } catch(e) {}
}

function init() {
  document.getElementById('st-n-stocks').textContent = '{{ n_stocks }} 只';
  document.getElementById('st-n-rows').textContent = '{{ n_rows }} 行';
  document.getElementById('st-n-ind').textContent = '{{ n_industries }} 个';
  document.getElementById('st-latest').textContent = '{{ sync_latest_trading }}';
  document.getElementById('st-sync-detail').textContent = '{{ sync_message }}';
  vsInit();
  vsSetStocks(allStocks);
  setInterval(refreshSyncDot, 8000);
  refreshSyncDot();
  selectStock('600519');
  showToast('{{ sync_message }}');
  pollSyncStatus();
  refreshMarketBar();
  loadCachedPicks();
}

// 虚拟滚动 — 只渲染可见区域，大幅减少 DOM 节点
const VS_H = 72, VS_BUF = 8;
let vsStocks = [];
let sortMode = localStorage.getItem('ql-sort') || 'code';
let sortDir = parseInt(localStorage.getItem('ql-sort-dir') || '1');

function applySort(stocks) {
  const dir = sortDir;
  if (sortMode === 'pct') {
    stocks.sort((a, b) => ((b.pct_change || 0) - (a.pct_change || 0)) * dir);
  } else if (sortMode === 'name') {
    stocks.sort((a, b) => (a.name || '').localeCompare(b.name || '', 'zh') * dir);
  } else {
    stocks.sort((a, b) => (a.code || '').localeCompare(b.code || '') * dir);
  }
  return stocks;
}

function setSort(mode) {
  if (sortMode === mode) { sortDir *= -1; }
  else { sortMode = mode; sortDir = (mode === 'pct') ? 1 : 1; }
  localStorage.setItem('ql-sort', sortMode);
  localStorage.setItem('ql-sort-dir', sortDir);
  updateSortBtns();
  vsSetStocks(applySort([...allStocks]));
  if (typeof loadWatchlist === 'function') loadWatchlist();
}

function updateSortBtns() {
  document.querySelectorAll('.sort-btn').forEach(b => {
    b.classList.remove('active');
    if (b.dataset.sort === sortMode) {
      b.classList.add('active');
      b.textContent = b.dataset.label + (sortDir > 0 ? '↑' : '↓');
    } else {
      b.textContent = b.dataset.label;
    }
  });
}

function vsInit() {
  const list = document.getElementById('stock-list');
  list.innerHTML = '<div class="vs-content"></div>';
  list.addEventListener('scroll', () => requestAnimationFrame(vsRender), {passive: true});
  list.addEventListener('click', e => {
    const item = e.target.closest('.stock-item');
    if (!item) return;
    if (e.target.closest('[data-action="wl"]')) {
      toggleWL(item.dataset.code, e.target.closest('[data-action="wl"]'));
      return;
    }
    selectStock(item.dataset.code);
  });
  list.addEventListener('dblclick', e => {
    const item = e.target.closest('.stock-item');
    if (item) toggleCompare(item.dataset.code);
  });
}

function vsRender() {
  const list = document.getElementById('stock-list');
  if (!list) return;
  const st = list.scrollTop, vh = list.clientHeight || 400;
  const total = vsStocks.length;
  const content = list.querySelector('.vs-content');
  if (!content) return;

  if (!total) {
    content.style.paddingTop = '0px';
    content.style.paddingBottom = '0px';
    content.innerHTML = '<div style="color:#484f58;padding:20px;text-align:center">无匹配结果</div>';
    return;
  }

  const start = Math.max(0, Math.floor(st / VS_H) - VS_BUF);
  const end = Math.min(total, Math.ceil((st + vh) / VS_H) + VS_BUF);
  // 用上下 padding 撑出完整滚动高度，只渲染可见区域
  content.style.paddingTop = (start * VS_H) + 'px';
  content.style.paddingBottom = ((total - end) * VS_H) + 'px';

  let h = '';
  for (let i = start; i < end; i++) {
    const s = vsStocks[i];
    const inWL = wlCodes.has(s.code);
	    // 数据新鲜度指示器
	    const today = new Date().toISOString().slice(0,10);
	    const lastDate = s.last_date || '';
	    const diffDays = lastDate ? Math.round((new Date(today) - new Date(lastDate)) / 86400000) : 999;
	    let dotColor = '#484f58', dotTitle = '无数据日期';
	    if (lastDate && diffDays <= 1) { dotColor = '#3fb950'; dotTitle = '最新: ' + lastDate; }
	    else if (lastDate && diffDays <= 3) { dotColor = '#f0883e'; dotTitle = '延迟: ' + lastDate + ' (' + diffDays + '天前)'; }
	    else if (lastDate && diffDays <= 7) { dotColor = '#f85149'; dotTitle = '过期: ' + lastDate + ' (' + diffDays + '天前)'; }
	    else if (lastDate) { dotColor = '#b0b0b0'; dotTitle = '严重过期: ' + lastDate + ' (' + diffDays + '天前)'; }
	    h += '<div class="stock-item ' + (s.code===currentCode?'active':'') + '" data-code="' + s.code + '" style="height:' + (VS_H-1) + 'px">'
	      + '<div class="info"><span class="code"><span class="fresh-dot" style="background:' + dotColor + '" title="' + dotTitle + '"></span>' + s.code + ' <span style="font-weight:400;font-size:11px;color:#8b949e">' + (s.industry||'') + '</span></span>'
	      + '<span class="name">' + (s.name||'') + '</span></div>'
	      + '<div class="right"><span class="ret ' + ((s.pct_change||0)>=0?'pos':'neg') + '">' + ((s.pct_change||0)>=0?'+':'') + (s.pct_change||0).toFixed(2) + '%</span>'
	      + '<span class="extra">PE ' + (s.pe||'--') + ' | ROE ' + (s.roe||'--') + '%</span>'
	      + '<button class="wl-add-btn ' + (inWL?'added':'') + '" data-action="wl">' + (inWL?'已添加':'+自选') + '</button></div></div>';
  }
  content.innerHTML = h;
}

function vsUpdateActive() {
  document.querySelectorAll('#stock-list .stock-item').forEach(el => {
    el.classList.toggle('active', el.dataset.code === currentCode);
  });
}

function vsUpdateWLButtons() {
  document.querySelectorAll('#stock-list [data-action="wl"]').forEach(btn => {
    const item = btn.closest('.stock-item');
    if (!item) return;
    const inWL = wlCodes.has(item.dataset.code);
    btn.textContent = inWL ? '已添加' : '+自选';
    btn.classList.toggle('added', inWL);
  });
}

function vsSetStocks(stocks) {
  vsStocks = applySort([...stocks]);
  updateSortBtns();
  const list = document.getElementById('stock-list');
  if (list) list.scrollTop = 0;
  vsRender();
}

let filterDebounce = null;
function filterStocks() {
  clearTimeout(filterDebounce);
  filterDebounce = setTimeout(() => {
    const q = document.getElementById('search').value.toLowerCase();
    vsSetStocks(applySort([...allStocks]).filter(s => (s.code+s.name+s.industry).toLowerCase().includes(q)));
  }, 200);
}

var _selectReqId = 0;
async function selectStock(code) {
  currentCode = code;
  vsUpdateActive();
  document.getElementById('search').value = '';
  var reqId = ++_selectReqId;
  var resp = await fetch('/api/chart/'+code);
  if (reqId !== _selectReqId) return; // 过期请求，丢弃
  const data = await resp.json();
  renderMetrics(data.stats);
  [data.price, data.ret_hist, data.volume, data.valuation, data.roe_chart, data.dupont_chart].forEach(fig => {
    if (fig.layout.template) fig.layout.template = null;
    fig.layout.paper_bgcolor = plotlyPaperBg;
    fig.layout.plot_bgcolor = plotlyPlotBg;
  });
  const plotCfg = {transition:{duration:300,easing:'cubic-in-out'}};
  Plotly.react('chart-price', data.price, plotCfg, {responsive:true});
  Plotly.react('chart-ret', data.ret_hist, plotCfg, {responsive:true});
  Plotly.react('chart-vol', data.volume, plotCfg, {responsive:true});
  Plotly.react('chart-val', data.valuation, plotCfg, {responsive:true});
  Plotly.react('chart-roe', data.roe_chart, plotCfg, {responsive:true});
  Plotly.react('chart-dupont', data.dupont_chart, plotCfg, {responsive:true});
}

function toggleCompare(code) {
  if(code===currentCode)return;
  compareCode = compareCode===code ? null : code;
  showToast(compareCode ? '对比: '+currentCode+' vs '+compareCode : '取消对比');
  if(compareCode)loadCompare(); else selectStock(currentCode);
}

async function loadCompare() {
  const data = await (await fetch('/api/compare?a='+currentCode+'&b='+compareCode)).json();
  if (data.price.layout.template) data.price.layout.template = null;
  data.price.layout.paper_bgcolor = plotlyPaperBg;
  data.price.layout.plot_bgcolor = plotlyPlotBg;
  Plotly.react('chart-price', data.price, {transition:{duration:300,easing:'cubic-in-out'}}, {responsive:true});
}

function renderMetrics(s) {
  const fc = (v) => v != null ? v : '--';
  const color = s.pct_change >= 0 ? '#f85149' : '#3fb950';

  // 价格头部
  document.getElementById('metrics-price').innerHTML = `
    <div class="price-header">
      <span class="ph-name">${s.name||s.code} <span style="font-weight:400;font-size:13px;color:#8b949e">${s.code}</span></span>
      <span class="ph-price" style="color:${color}">${s.latest}</span>
      <span class="ph-pct" style="color:${color}">${s.pct_change>=0?'+':''}${s.pct_change}%</span>
      <span class="ph-date">📅 ${s.last_data_date} ${s.last_data_date < new Date().toISOString().slice(0,10) ? '⚠延迟' : ''}</span>
    </div>`;

  // 7列2行指标
  document.getElementById('metrics-fin').innerHTML = `
    <div class="detail-grid">
      <div class="dg-item"><span class="dg-label">今开</span><span class="dg-val">${fc(s.open)}</span></div>
      <div class="dg-item"><span class="dg-label">最高</span><span class="dg-val" style="color:#f85149">${fc(s.high)}</span></div>
      <div class="dg-item"><span class="dg-label">涨停</span><span class="dg-val" style="color:#f85149">${fc(s.limit_up)}</span></div>
      <div class="dg-item"><span class="dg-label">换手</span><span class="dg-val">${fc(s.turnover)}%</span></div>
      <div class="dg-item"><span class="dg-label">成交量</span><span class="dg-val">${fc(s.volume_str)}</span></div>
      <div class="dg-item"><span class="dg-label">市盈(动)</span><span class="dg-val">${fc(s.pe)}</span></div>
      <div class="dg-item"><span class="dg-label">总市值</span><span class="dg-val">${fc(s.total_cap)}</span></div>

      <div class="dg-item"><span class="dg-label">昨收</span><span class="dg-val">${fc(s.pre_close)}</span></div>
      <div class="dg-item"><span class="dg-label">最低</span><span class="dg-val" style="color:#3fb950">${fc(s.low)}</span></div>
      <div class="dg-item"><span class="dg-label">跌停</span><span class="dg-val" style="color:#3fb950">${fc(s.limit_down)}</span></div>
      <div class="dg-item"><span class="dg-label">量比</span><span class="dg-val">${fc(s.vol_ratio)}</span></div>
      <div class="dg-item"><span class="dg-label">成交额</span><span class="dg-val">${fc(s.amount_str)}</span></div>
      <div class="dg-item"><span class="dg-label">市净</span><span class="dg-val">${fc(s.pb)}</span></div>
      <div class="dg-item"><span class="dg-label">流通市值</span><span class="dg-val">${fc(s.circ_cap)}</span></div>
    </div>`;

  // 财务摘要（如果有）
  if (s.roe != null) {
    document.getElementById('metrics-fin').innerHTML += `
    <div class="fin-summary">
      <span>ROE ${s.roe}%</span>
      <span>净利率 ${s.np_margin||'--'}%</span>
      <span>负债率 ${s.debt_ratio||'--'}%</span>
      <span>净利增速 ${(s.yoy_ni||0)>=0?'+':''}${s.yoy_ni||'--'}%</span>
    </div>`;
  }
}

// === 同步状态轮询 ===
async function pollSyncStatus() {
  try {
    const r = await fetch('/api/sync-status');
    const s = await r.json();
    const dot = document.getElementById('sync-dot');
    const text = document.getElementById('sync-text');
    if (s.running) {
      dot.className = 'dot syncing';
      text.textContent = s.message + ' (' + s.start_time + ')';
      setTimeout(pollSyncStatus, 5000);
    } else {
      dot.className = 'dot ok';
      text.textContent = s.message;
      setTimeout(pollSyncStatus, 60000);
    }
  } catch(e) { setTimeout(pollSyncStatus, 10000); }
}

// === 同步详情页 ===
const PHASE_ORDER = ['names', 'industries', 'daily', 'financial', 'done'];

async function pollSyncDetail() {
  try {
    const r = await fetch('/api/sync-status');
    const s = await r.json();

    // 更新阶段指示器
    let foundActive = false;
    PHASE_ORDER.forEach(p => {
      const el = document.getElementById('phase-'+p);
      if(!el) return;
      el.classList.remove('done','active');
      if(p === s.phase) { el.classList.add('active'); foundActive = true; }
      else if(!foundActive) { el.classList.add('done'); }
    });

    // 当前股票和类型
    const code = s.current_stock || '';
    const name = s.current_stock_name || code;
    const typeNames = {daily:'日线数据', financial:'财务数据'};
    const typeName = typeNames[s.current_type] || '';
    if (code && typeName) {
      document.getElementById('cur-stock').textContent = '正在同步【' + name + '】的【' + typeName + '】中...';
    } else if (s.phase === 'daily_financial') {
      document.getElementById('cur-stock').textContent = '日线和财务数据并行同步中...';
    } else {
      document.getElementById('cur-stock').textContent = '等待中...';
    }
    document.getElementById('cur-type').textContent = code || '--';

    // 日线进度条
    const dTotal = s.daily_total || 0;
    const dDone = s.daily_done || 0;
    const dPct = dTotal > 0 ? Math.round(dDone/dTotal*100) : (s.completed ? 100 : 0);
    document.getElementById('daily-bar').style.width = dPct+'%';
    document.getElementById('daily-bar').className = 'progress-bar-fill' + (s.phase==='daily'?' running':'');
    document.getElementById('daily-text').textContent = dTotal>0 ? (dDone+' / '+dTotal) : (s.completed?'已完成':'等待中...');
    document.getElementById('daily-pct').textContent = dPct+'%';

    // 财务进度条
    const fTotal = s.fin_total || 0;
    const fDone = s.fin_done || 0;
    const fPct = fTotal > 0 ? Math.round(fDone/fTotal*100) : (s.completed ? 100 : 0);
    document.getElementById('fin-bar').style.width = fPct+'%';
    document.getElementById('fin-bar').className = 'progress-bar-fill' + (s.phase==='financial'?' running':'');
    document.getElementById('fin-text').textContent = fTotal>0 ? (fDone+' / '+fTotal) : (s.completed?'已完成':'等待中...');
    document.getElementById('fin-pct').textContent = fPct+'%';

    // 日志
    const log = document.getElementById('sync-log');
    if(s.running && !s.completed) {
      const line = document.createElement('div');
      const now = new Date().toLocaleTimeString();
      const phaseName = {'names':'股票名称','industries':'行业分类','daily':'日线','financial':'财务','done':'完成'}[s.phase]||'';
      line.textContent = '['+now+'] ['+phaseName+'] '+s.message;
      log.appendChild(line);
      if(log.children.length > 80) log.removeChild(log.firstChild);
      log.scrollTop = log.scrollHeight;
    }
    if(!s.running && s.completed) {
      const existing = log.querySelector('.log-ok');
      if(!existing) {
        const line = document.createElement('div');
        line.className = 'log-ok';
        line.textContent = '['+new Date().toLocaleTimeString()+'] '+s.message;
        log.appendChild(line);
        log.scrollTop = log.scrollHeight;
      }
    }

    // 更新顶部横幅
    const dot = document.getElementById('sync-dot');
    const text = document.getElementById('sync-text');
    if(s.running) {
      dot.className = 'dot syncing';
      text.textContent = s.message + ' (' + (s.start_time||'') + ')';
    } else if(s.completed) {
      dot.className = 'dot ok';
      text.textContent = s.message;
    }
  } catch(e) {}
}

async function loadDataSummary() {
  try {
    const r = await fetch('/api/data/summary');
    const s = await r.json();
    document.getElementById('ss-daily-range').textContent = s.daily_earliest || '--';
    document.getElementById('ss-daily-count').textContent = s.daily_count + ' 只';
    document.getElementById('ss-fin-range').textContent = s.financial_earliest || '--';
    document.getElementById('ss-fin-count').textContent = s.financial_count + ' 只';
    if(s.daily_latest) {
      document.getElementById('ss-daily-range').textContent += ' ~ ' + s.daily_latest;
    }
    if(s.financial_latest) {
      document.getElementById('ss-fin-range').textContent += ' ~ ' + s.financial_latest;
    }
  } catch(e) {}
}

async function triggerManualSync() {
  const btn = document.getElementById('btn-manual-sync');
  const msg = document.getElementById('sync-action-msg');
  btn.disabled = true;
  btn.textContent = '启动中...';
  msg.textContent = '';
  try {
    const r = await fetch('/api/sync/trigger', {method:'POST',headers:{'Content-Type':'application/json'}});
    const d = await r.json();
    if(d.ok) {
      msg.textContent = d.message;
      msg.style.color = '#3fb950';
      pollSyncDetail();
      syncPollTimer = setInterval(pollSyncDetail, 3000);
    } else {
      msg.textContent = d.message;
      msg.style.color = '#f85149';
    }
  } catch(e) {
    msg.textContent = '请求失败';
    msg.style.color = '#f85149';
  }
  btn.disabled = false;
  btn.textContent = '手动拉取最新';
}

async function triggerHistoricalSync() {
  const start = document.getElementById('hist-start').value;
  const end = document.getElementById('hist-end').value;
  const msg = document.getElementById('hist-msg');
  if(!start || !end) { msg.textContent = '请选择起止日期'; msg.style.color = '#f85149'; return; }
  if(start > end) { msg.textContent = '开始日期不能晚于结束日期'; msg.style.color = '#f85149'; return; }
  msg.textContent = '正在启动...';
  msg.style.color = '#f0883e';
  try {
    const r = await fetch('/api/sync/historical', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start_date:start,end_date:end})});
    const d = await r.json();
    if(d.ok) {
      msg.textContent = d.message;
      msg.style.color = '#3fb950';
      pollSyncDetail();
      syncPollTimer = setInterval(pollSyncDetail, 3000);
    } else {
      msg.textContent = d.message;
      msg.style.color = '#f85149';
    }
  } catch(e) {
    msg.textContent = '请求失败';
    msg.style.color = '#f85149';
  }
}

function showToast(m) {
  const t=document.getElementById('toast');t.textContent=m;t.style.display='block';
  setTimeout(()=>t.style.display='none',2000);
}

// ============================================================
// 认证
// ============================================================
let authMode = 'login';

async function checkAuthState() {
  try {
    const r = await fetch('/api/auth/me');
    const d = await r.json();
    if(d.logged_in) {
      _loggedIn = true;
      document.getElementById('auth-guest').style.display = 'none';
      document.getElementById('auth-logged').style.display = '';
      document.getElementById('auth-username').textContent = d.user.username;
      if(d.user.is_admin) document.getElementById('auth-username').textContent += ' [管理员]';
      // 显示管理员专属标签页按钮
      if(d.user.is_admin) {
        document.getElementById('tab-btn-admin').style.display = '';
        document.getElementById('tab-btn-sync').style.display = '';
      }
      loadWLCodes().then(() => vsUpdateWLButtons());
    }
  } catch(e) {}
}

function showLoginModal() {
  authMode = 'login';
  document.getElementById('modal-title').textContent = '登录';
  document.getElementById('modal-submit-btn').textContent = '登录';
  document.getElementById('modal-switch-text').textContent = '没有账号？';
  document.getElementById('modal-switch-link').textContent = '去注册';
  document.getElementById('modal-error').textContent = '';
  document.getElementById('login-modal').style.display = 'flex';
}

function showRegisterModal() {
  authMode = 'register';
  document.getElementById('modal-title').textContent = '注册';
  document.getElementById('modal-submit-btn').textContent = '注册';
  document.getElementById('modal-switch-text').textContent = '已有账号？';
  document.getElementById('modal-switch-link').textContent = '去登录';
  document.getElementById('modal-error').textContent = '';
  document.getElementById('login-modal').style.display = 'flex';
}

function hideLoginModal() {
  document.getElementById('login-modal').style.display = 'none';
}

function toggleAuthMode() {
  if(authMode === 'login') showRegisterModal();
  else showLoginModal();
}

async function handleAuthSubmit() {
  const u = document.getElementById('modal-username').value.trim();
  const p = document.getElementById('modal-password').value;
  const err = document.getElementById('modal-error');
  if(!u||!p) { err.textContent = '请填写用户名和密码'; return; }

  const url = authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
  try {
    const r = await fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})});
    const d = await r.json();
    if(d.success) {
      hideLoginModal();
      document.getElementById('modal-username').value = '';
      document.getElementById('modal-password').value = '';
      checkAuthState();
      if(authMode === 'register') showToast('注册成功！请登录');
      else { showToast('登录成功'); loadWatchlist(); loadStrategyTab(); }
    } else {
      err.textContent = d.message || '操作失败';
    }
  } catch(e) { err.textContent = '网络错误'; }
}

async function handleLogout() {
  await fetch('/api/auth/logout', {method:'POST',headers:{'Content-Type':'application/json'}});
  _loggedIn = false;
  document.getElementById('auth-guest').style.display = '';
  document.getElementById('auth-logged').style.display = 'none';
  document.getElementById('tab-btn-admin').style.display = 'none';
  document.getElementById('tab-btn-sync').style.display = 'none';
  // 重置策略页为访客视图
  loadStrategyTab();
  // 如果当前在需要登录的页，跳回仪表盘
  if(activeTab==='sync') switchTab('dash');
  wlCodes = new Set();
  vsUpdateWLButtons();
  loadWatchlist();
  showToast('已登出');
}

// ============================================================
// 自选股
// ============================================================

async function loadWatchlist() {
  try {
    const r = await fetch('/api/auth/me');
    const me = await r.json();
    if(!me.logged_in) {
      document.getElementById('wl-guest-view').style.display = '';
      document.getElementById('wl-user-view').style.display = 'none';
      return;
    }
    document.getElementById('wl-guest-view').style.display = 'none';
    document.getElementById('wl-user-view').style.display = 'flex';

    const wl = await (await fetch('/api/watchlist')).json();
    document.getElementById('wl-count').textContent = wl.length;
    const list = document.getElementById('wl-list');
    if(!wl.length) {
      list.innerHTML = '<div style="color:#484f58;font-size:13px;padding:20px">暂无自选股，在左侧搜索添加</div>';
      return;
    }
    // 按当前排序模式排序
    wl.sort((a, b) => {
      if (sortMode === 'pct') return ((b.pct_change || 0) - (a.pct_change || 0)) * sortDir;
      if (sortMode === 'name') return (a.name || '').localeCompare(b.name || '', 'zh') * sortDir;
      return (a.code || '').localeCompare(b.code || '') * sortDir;
    });
    list.innerHTML = wl.map(s => `<div class="wl-item" onclick="viewWatchStock('${s.code}')" style="cursor:pointer">
      <span class="code">${s.code}</span><span class="name">${s.name||''}</span>
      <span style="color:${(s.pct_change||0)>=0?'var(--red)':'var(--green)'}" class="extra">${s.latest?'¥'+s.latest:''} ${s.pct_change!=null?(s.pct_change>=0?'+':'')+s.pct_change+'%':''}</span>
      <button class="btn-danger" onclick="event.stopPropagation();removeFromWatchlist('${s.code}')">删除</button>
    </div>`).join('');
  } catch(e) {}
}

async function searchWatchStocks() {
  const q = document.getElementById('wl-search').value.trim().toLowerCase();
  const results = document.getElementById('wl-search-results');
  if(!q) { results.innerHTML = ''; return; }

  // 从 allStocks 中搜索匹配的股票
  const matches = allStocks.filter(s =>
    (s.code + s.name + (s.industry||'')).toLowerCase().includes(q)
  ).slice(0, 10);

  results.innerHTML = matches.map(s => `<div class="wl-item">
    <span class="code">${s.code}</span><span class="name">${s.name||''}</span>
    <span class="extra">${s.industry||''} ${s.pct_change!=null?(s.pct_change>=0?'+':'')+s.pct_change.toFixed(2)+'%':''}</span>
    <button class="btn-sm" onclick="addToWatchlist('${s.code}')">+添加</button>
  </div>`).join('');

  if(!matches.length && q) results.innerHTML = '<div style="color:#484f58;padding:8px">未找到匹配股票</div>';
}

async function loadWLCodes() {
  try {
    const r = await fetch('/api/watchlist');
    const wl = await r.json();
    wlCodes = new Set(wl.map(s => s.code));
  } catch(e) { wlCodes = new Set(); }
}

async function toggleWL(code, btn) {
  if (wlCodes.has(code)) {
    const r = await fetch('/api/watchlist/remove', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});
    const d = await r.json();
    if (d.success) { wlCodes.delete(code); btn.textContent = '+自选'; btn.classList.remove('added'); showToast('已取消自选'); }
  } else {
    const r = await fetch('/api/watchlist/add', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});
    const d = await r.json();
    if (d.success) { wlCodes.add(code); btn.textContent = '已添加'; btn.classList.add('added'); showToast('已添加自选'); }
  }
}

async function addToWatchlist(code) {
  try {
    const r = await fetch('/api/watchlist/add', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});
    const d = await r.json();
    showToast(d.message);
    if(d.success) { wlCodes.add(code); vsUpdateWLButtons(); loadWatchlist(); searchWatchStocks(); }
  } catch(e) { showToast('操作失败'); }
}

async function removeFromWatchlist(code) {
  try {
    const r = await fetch('/api/watchlist/remove', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});
    const d = await r.json();
    showToast(d.message);
    if(d.success) { wlCodes.delete(code); vsUpdateWLButtons(); loadWatchlist(); }
  } catch(e) { showToast('操作失败'); }
}

async function viewWatchStock(code) {
  document.getElementById('wl-detail-empty').style.display = 'none';
  document.getElementById('wl-detail-content').style.display = '';
  try {
    const data = await (await fetch('/api/chart/' + code)).json();
    renderMetrics(data.stats);
    document.getElementById('wl-metrics-price').innerHTML = document.getElementById('metrics-price').innerHTML;
    document.getElementById('wl-metrics-fin').innerHTML = document.getElementById('metrics-fin').innerHTML;
    [data.price, data.ret_hist, data.volume, data.valuation, data.roe_chart, data.dupont_chart].forEach(fig => {
      if (fig.layout.template) fig.layout.template = null;
      fig.layout.paper_bgcolor = plotlyPaperBg;
      fig.layout.plot_bgcolor = plotlyPlotBg;
    });
    const pCfg = {transition:{duration:300,easing:'cubic-in-out'}};
    Plotly.react('wl-chart-price', data.price, pCfg, {responsive:true});
    Plotly.react('wl-chart-ret', data.ret_hist, pCfg, {responsive:true});
    Plotly.react('wl-chart-vol', data.volume, pCfg, {responsive:true});
    Plotly.react('wl-chart-val', data.valuation, pCfg, {responsive:true});
    Plotly.react('wl-chart-roe', data.roe_chart, pCfg, {responsive:true});
    Plotly.react('wl-chart-dupont', data.dupont_chart, pCfg, {responsive:true});
  } catch(e) { showToast('加载失败'); }
}

// ============================================================
// 管理员
// ============================================================

async function loadAdminUsers() {
  try {
    const users = await (await fetch('/api/admin/users')).json();
    const div = document.getElementById('admin-users-table');
    div.innerHTML = '<div style="color:#8b949e;margin-bottom:6px">用户列表</div>' +
      users.map(u => `<div class="admin-row">
        <span class="uname">${u.username}${u.is_admin?' [管理员]':''}</span>
        <span class="meta">自选:${u.watchlist_count} | ${u.created_at||''}</span>
        <span>
          <button class="btn-sm" style="font-size:10px;padding:2px 8px" onclick="viewUserWatchlist(${u.id},'${u.username}')">查看</button>
          ${u.username!=='admin'?`<button class="btn-danger" style="margin-left:4px" onclick="adminDeleteUser(${u.id})">删除</button>`:''}
        </span>
      </div>`).join('');
  } catch(e) {}
}

async function adminAddUser() {
  const u = document.getElementById('admin-new-user').value.trim();
  const p = document.getElementById('admin-new-pass').value;
  const isAdmin = document.getElementById('admin-new-admin').checked;
  if(!u||!p) { showToast('请填写用户名和密码'); return; }
  try {
    const r = await fetch('/api/admin/add-user', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p,is_admin:isAdmin})});
    const d = await r.json();
    showToast(d.message);
    if(d.success) { loadAdminUsers(); document.getElementById('admin-new-user').value=''; document.getElementById('admin-new-pass').value=''; }
  } catch(e) { showToast('操作失败'); }
}

async function adminDeleteUser(uid) {
  if(!confirm('确认删除该用户？')) return;
  try {
    const r = await fetch('/api/admin/delete-user', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:uid})});
    const d = await r.json();
    showToast(d.message);
    if(d.success) loadAdminUsers();
  } catch(e) { showToast('操作失败'); }
}

async function viewUserWatchlist(uid, uname) {
  try {
    const wl = await (await fetch('/api/admin/user-watchlist/' + uid)).json();
    const div = document.getElementById('admin-user-wl');
    if(!wl.length) { div.innerHTML = `<div style="color:#8b949e">${escHtml(uname)} 暂无自选股</div>`; return; }
    div.innerHTML = `<div style="color:#8b949e;margin-bottom:4px">${escHtml(uname)} 的自选股:</div>` +
      wl.map(s => `<span style="color:#58a6ff;margin-right:8px">${escHtml(s.code)} ${escHtml(s.name||'')}</span>`).join('');
  } catch(e) {}
}

// ============================================================
// 纸交易
// ============================================================

function showStatsModal() {
  document.getElementById('stats-modal').style.display = 'flex';
}
function hideStatsModal() { document.getElementById('stats-modal').style.display = 'none'; }
async function refreshSyncDot() {
  try {
    const r = await fetch('/api/sync-status');
    if (!r.ok) return;
    const d = await r.json();
    if (d.running) {
      document.getElementById('st-sync').textContent = '同步中';
      document.getElementById('st-sync').style.color = 'var(--orange)';
    } else if (d.completed) {
      document.getElementById('st-sync').textContent = '已完成';
      document.getElementById('st-sync').style.color = 'var(--green)';
    }
  } catch(e) {}
}

async function loadCachedPicks() {
  try {
    const r = await fetch('/api/strategy/recommend/cached');
    const d = await r.json();
    if (d.success && d.picks && d.picks.length > 0) {
      renderPicks(d);
    }
  } catch(e) {}
}

function loadStrategyTab() {
  const guest = document.getElementById('strategy-guest-view');
  const user = document.getElementById('strategy-user-view');
  if (!_loggedIn) {
    if (guest) guest.style.display = '';
    if (user) user.style.display = 'none';
    return;
  }
  if (guest) guest.style.display = 'none';
  if (user) user.style.display = '';
  refreshAccount();
  if (_cachedPicks) renderPicks({picks: _cachedPicks, success: true, target_date: _cachedPicksDate});
}

async function refreshPicks() {
  const div = document.getElementById('strategy-picks');
  const progWrap = document.getElementById('rec-progress-wrap');
  const progBar = document.getElementById('rec-progress-bar');
  const progPct = document.getElementById('rec-progress-pct');
  const progStep = document.getElementById('rec-progress-step');
  const progMsg = document.getElementById('rec-progress-msg');
  const btn = document.getElementById('btn-refresh-picks');

  btn.disabled = true;
  btn.textContent = '计算中...';
  progWrap.style.display = 'block';
  div.innerHTML = '';

  try {
    // 启动异步推荐
    const startR = await fetch('/api/strategy/recommend/start', { method: 'POST', headers: {'Content-Type': 'application/json'} });
    const startD = await startR.json();
    if (!startD.success) {
      progWrap.style.display = 'none';
      div.innerHTML = '<span style="color:var(--red)">' + startD.message + '</span>';
      btn.disabled = false;
      btn.textContent = '刷新推荐';
      return;
    }

    // 轮询进度
    const stepLabels = { load: '加载数据', factors: '计算因子', ic: 'IC分析', score: '因子打分', build: '组合构建', reason: '生成理由', done: '完成' };

    for (let i = 0; i < 180; i++) {
      await new Promise(r => setTimeout(r, 1000));
      if (activeTab !== 'strategy') break; // 用户已切走，停止轮询
      const sR = await fetch('/api/strategy/recommend/status');
      const s = await sR.json();

      progBar.style.width = s.progress + '%';
      progPct.textContent = s.progress + '%';
      progStep.textContent = stepLabels[s.step] || s.step || '处理中';
      progMsg.textContent = s.message || '';

      if (!s.running) {
        if (s.error) {
          progWrap.style.display = 'none';
          div.innerHTML = '<span style="color:var(--red)">推荐失败: ' + s.error + '</span>';
        } else if (s.result) {
          renderPicks(s.result);
          progWrap.style.display = 'none';
        }
        break;
      }
    }
    if (progWrap.style.display !== 'none') {
      progWrap.style.display = 'none';
      div.innerHTML = '<span style="color:var(--orange)">推荐计算超时，请重试</span>';
    }
  } catch (e) {
    progWrap.style.display = 'none';
    div.innerHTML = '<span style="color:var(--red)">请求失败</span>';
  }
  btn.disabled = false;
  btn.textContent = '刷新推荐';
}

function renderPicks(d) {
  const div = document.getElementById('strategy-picks');
  let h = '<div style="font-size:11px;color:var(--text2);margin-bottom:8px">数据日期: ' + d.target_date + ' | 共 ' + d.picks.length + ' 只推荐</div>';

  // 风格标签颜色
  const styleColors = { '防御型': '#4a9', '进攻型': '#f6a', '均衡型': '#58a6ff' };

  d.picks.forEach((p, i) => {
    const styleColor = styleColors[p.style] || '#58a6ff';

    // 因子标签
    let factorTags = '';
    if (p.top_factors && p.top_factors.length > 0) {
      factorTags = '<div class="factor-tags">';
      p.top_factors.forEach(f => {
        const cls = f.direction === '看好' ? 'factor-tag-good' : 'factor-tag-bad';
        factorTags += '<span class="factor-tag ' + cls + '" title="' + f.name + ' Z=' + f.z_score.toFixed(2) + ' 权重=' + (f.weight*100).toFixed(1) + '%">' + f.name + ' ' + (f.z_score>=0?'+':'') + f.z_score.toFixed(1) + '</span>';
      });
      factorTags += '</div>';
    }

    // 排名徽章颜色
    let rankBadgeColor = 'var(--text2)';
    if (p.percentile >= 90) rankBadgeColor = 'var(--green)';
    else if (p.percentile >= 75) rankBadgeColor = 'var(--accent)';
    else if (p.percentile >= 50) rankBadgeColor = 'var(--orange)';

    h += '<div class="pick-card">'
      + '<div class="pick-card-header">'
        + '<span class="pick-rank">#' + (i + 1) + '</span>'
        + '<span class="pick-code-name">' + p.code + ' <strong>' + p.name + '</strong></span>'
        + '<span class="pick-style-badge" style="background:' + styleColor + '20;color:' + styleColor + ';border:1px solid ' + styleColor + '40">' + (p.style || p.cluster) + '</span>'
        + '<span class="pick-percentile" style="color:' + rankBadgeColor + '">超越 ' + p.percentile + '%</span>'
      + '</div>'
      + '<div class="pick-card-body">'
        + '<div class="pick-meta-row">'
          + '<span class="pick-meta-item">行业: ' + p.industry + '</span>'
          + '<span class="pick-meta-item">收盘: ¥' + p.close.toFixed(2) + '</span>'
          + '<span class="pick-meta-item">得分: ' + p.score.toFixed(3) + '</span>'
          + '<span class="pick-meta-item">权重: ' + (p.weight * 100).toFixed(1) + '%</span>'
          + '<span class="pick-meta-item">' + (p.cluster_rank || '') + '</span>'
        + '</div>'
        + '<div class="pick-reason">"' + (p.reason || p.strategy_desc || '') + '"</div>'
        + factorTags
      + '</div>'
      + '<div class="pick-card-footer">'
        + '<button class="btn-xs" onclick="pickBuy(\'' + p.code + '\',' + p.close + ')">买入</button>'
      + '</div>'
      + '</div>';
  });
  div.innerHTML = h;
  // 构建名称映射和缓存
  window._pickNameMap = {};
  d.picks.forEach(p => { window._pickNameMap[p.code] = p.name; });
  window._cachedPicks = d.picks;
  window._cachedPicksDate = d.target_date;
  document.getElementById('btn-basket-buy').style.display = 'inline-block';
}

let _pickCode = '', _pickPrice = 0, _pickName = '';
function pickBuy(code, price) {
  _pickCode = code; _pickPrice = price;
  showBuyModal(code, price);
}
async function showBuyModal(code, price) {
  document.getElementById('bm-code').textContent = code;
  const nm = (window._pickNameMap && window._pickNameMap[code]) || _pickName || code;
  document.getElementById('bm-name').textContent = nm;
  document.getElementById('bm-price').textContent = '¥' + price.toFixed(2);
  document.getElementById('bm-shares').value = 100;
  // 获取可用资金
  try {
    const r = await fetch('/api/trade/account');
    const d = await r.json();
    document.getElementById('bm-cash').textContent = '¥' + d.balance.toLocaleString();
    window._bmCash = d.balance;
  } catch(e) { window._bmCash = 0; }
  document.getElementById('bm-msg').textContent = '';
  document.getElementById('bm-msg').style.color = 'var(--green)';
  updateBuyCalc();
  document.getElementById('buy-modal').style.display = 'flex';
}
function hideBuyModal() { document.getElementById('buy-modal').style.display = 'none'; }
function updateBuyCalc() {
  const shares = parseInt(document.getElementById('bm-shares').value) || 100;
  const adj = Math.floor(shares / 100) * 100;
  if (adj !== shares) document.getElementById('bm-shares').value = adj;
  const cost = adj * _pickPrice * 1.001;
  const comm = Math.max(cost * 0.0003, 5);
  document.getElementById('bm-calc').innerHTML = '预计花费: ¥' + (cost + comm).toFixed(2)
    + ' | 佣金: ¥' + comm.toFixed(2)
    + ' | 滑点: ¥' + (cost - adj * _pickPrice).toFixed(2);
}
function setBuyShares(n) { document.getElementById('bm-shares').value = n; updateBuyCalc(); }
function setBuyMaxShares() {
  const price = _pickPrice;
  if (price <= 0 || !window._bmCash) return;
  const maxShares = Math.floor(window._bmCash / (price * 1.001 * 1.0003) / 100) * 100;
  document.getElementById('bm-shares').value = Math.max(100, maxShares);
  updateBuyCalc();
}
async function confirmBuy() {
  const shares = parseInt(document.getElementById('bm-shares').value) || 100;
  const adj = Math.floor(shares / 100) * 100;
  const msg = document.getElementById('bm-msg');
  if (adj < 100) { msg.textContent = '最少买入100股'; msg.style.color = 'var(--red)'; return; }
  msg.textContent = '交易中...';
  try {
    const r = await fetch('/api/trade/buy', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:_pickCode,shares:adj})});
    const d = await r.json();
    msg.textContent = d.message;
    msg.style.color = d.success ? 'var(--green)' : 'var(--red)';
    if (d.success) { refreshAccount(); setTimeout(hideBuyModal, 1000); }
  } catch(e) { msg.textContent = '交易失败'; msg.style.color = 'var(--red)'; }
}

// ---- 一键买入 ----
function showBasketModal() {
  if (!window._cachedPicks || !window._cachedPicks.length) { alert('请先生成推荐'); return; }
  fetch('/api/trade/account').then(r=>r.json()).then(d=>{
    document.getElementById('bk-cash').textContent = '¥' + d.balance.toLocaleString();
    window._bkCash = d.balance;
  });
  document.getElementById('bk-date').textContent = window._cachedPicksDate || '-';
  document.getElementById('bk-count').textContent = window._cachedPicks.length + ' 只';
  // 估算每只购买股数
  let rows = '';
  const totalWeight = window._cachedPicks.reduce((s,p) => s + (p.weight||0), 0);
  window._cachedPicks.forEach(p => {
    const w = (p.weight || 0) / Math.max(totalWeight, 0.001);
    const budget = (window._bkCash || 100000) * w;
    const estShares = Math.floor(budget / (p.close * 1.001) / 100) * 100;
    rows += '<div class="basket-row"><span>' + p.code + ' <strong>' + p.name + '</strong></span><span style="font-size:11px;color:var(--text2)">权重 ' + (w*100).toFixed(1) + '% | 约' + Math.max(100, estShares) + '股 | ~¥' + (Math.max(100, estShares)*p.close*1.001).toFixed(0) + '</span></div>';
  });
  document.getElementById('bk-list').innerHTML = rows || '<span style="color:var(--text2)">无推荐数据</span>';
  document.getElementById('bk-msg').textContent = '';
  document.getElementById('basket-modal').style.display = 'flex';
}
function hideBasketModal() { document.getElementById('basket-modal').style.display = 'none'; }
async function confirmBasketBuy() {
  const msg = document.getElementById('bk-msg');
  msg.textContent = '批量买入中...';
  msg.style.color = 'var(--accent)';
  try {
    const r = await fetch('/api/trade/buy_basket', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({picks:window._cachedPicks, total_cash:window._bkCash})});
    const d = await r.json();
    if (d.success) {
      msg.textContent = '成功买入 ' + d.bought + ' 只股票!';
      msg.style.color = 'var(--green)';
      refreshAccount();
      setTimeout(hideBasketModal, 1500);
    } else {
      msg.textContent = d.message || '买入失败';
      msg.style.color = 'var(--red)';
    }
  } catch(e) { msg.textContent = '网络错误'; msg.style.color = 'var(--red)'; }
}

// ---- 股票详情弹窗 ----
async function showDetailModal(code) {
  document.getElementById('detail-modal').style.display = 'flex';
  document.getElementById('dt-name').textContent = code;
  document.getElementById('dt-code-industry').textContent = '加载中...';
  document.getElementById('dt-market').innerHTML = '<span style="color:var(--text2)">加载中...</span>';
  document.getElementById('dt-fin').innerHTML = '<span style="color:var(--text2)">加载中...</span>';
  try {
    const r = await fetch('/api/stock/detail/' + code);
    const d = await r.json();
    if (!d.success) return;
    const s = d.stock;
    document.getElementById('dt-name').textContent = s.name + ' (' + s.code + ')';
    document.getElementById('dt-code-industry').textContent = '行业: ' + s.industry + ' | 最新数据: ' + s.last_date;
    const pctColor = s.pct_change >= 0 ? 'var(--red)' : 'var(--green)';
    document.getElementById('dt-market').innerHTML =
      '<div class="detail-item"><span class="lbl">最新价</span><span style="color:var(--red);font-weight:600">¥' + s.close + '</span></div>'
      + '<div class="detail-item"><span class="lbl">涨跌幅</span><span style="color:' + pctColor + ';font-weight:600">' + (s.pct_change>=0?'+':'') + s.pct_change + '%</span></div>'
      + '<div class="detail-item"><span class="lbl">成交量</span><span>' + (s.volume||'-') + ' 百万股</span></div>'
      + '<div class="detail-item"><span class="lbl">成交额</span><span>' + (s.amount||'-') + ' 亿元</span></div>'
      + '<div class="detail-item"><span class="lbl">PE(TTM)</span><span>' + (s.pe_ttm||'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">PB(MRQ)</span><span>' + (s.pb_mrq||'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">换手率</span><span>' + (s.turnover||'-') + '%</span></div>'
      + '<div class="detail-item"><span class="lbl">近1月收益</span><span style="color:' + ((s.ret_1m||0)>=0?'var(--red)':'var(--green)') + '">' + (s.ret_1m!=null?(s.ret_1m>=0?'+':'')+s.ret_1m+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">近3月收益</span><span style="color:' + ((s.ret_3m||0)>=0?'var(--red)':'var(--green)') + '">' + (s.ret_3m!=null?(s.ret_3m>=0?'+':'')+s.ret_3m+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">52周最高</span><span>¥' + (s.high_52w||'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">52周最低</span><span>¥' + (s.low_52w||'-') + '</span></div>';
    document.getElementById('dt-fin').innerHTML =
      '<div class="detail-item"><span class="lbl">ROE</span><span>' + (s.roe_avg!=null?s.roe_avg+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">净利润率</span><span>' + (s.np_margin!=null?s.np_margin+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">毛利率</span><span>' + (s.gp_margin!=null?s.gp_margin+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">净利润增速</span><span>' + (s.yoy_ni!=null?s.yoy_ni+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">净资产增速</span><span>' + (s.yoy_equity!=null?s.yoy_equity+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">流动比率</span><span>' + (s.current_ratio||'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">资产负债率</span><span>' + (s.liability_to_asset!=null?s.liability_to_asset+'%':'-') + '</span></div>'
      + '<div class="detail-item"><span class="lbl">资产周转率</span><span>' + (s.asset_turn_ratio||'-') + '</span></div>';
  } catch(e) { document.getElementById('dt-market').innerHTML = '<span style="color:var(--red)">加载失败</span>'; }
}
function hideDetailModal() { document.getElementById('detail-modal').style.display = 'none'; }

// ---- 卖出持仓 ----
async function sellPosition(code, maxShares) {
  const shares = prompt('卖出 ' + code + ' (最多 ' + maxShares + ' 股):', maxShares);
  if (!shares || parseInt(shares) < 100) return;
  try {
    const r = await fetch('/api/trade/sell', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:code,shares:parseInt(shares)})});
    const d = await r.json();
    alert(d.message);
    if (d.success) refreshAccount();
  } catch(e) { alert('交易失败'); }
}
async function sellAllPositions() {
  if (!confirm('确定要卖出所有持仓吗？此操作不可撤销。')) return;
  try {
    const r = await fetch('/api/trade/sell_all', {method:'POST',headers:{'Content-Type':'application/json'}});
    const d = await r.json();
    if (d.success) {
      const sold = d.results.filter(r=>r.success).length;
      alert('成功卖出 ' + sold + ' 只股票');
      refreshAccount();
    }
  } catch(e) { alert('操作失败'); }
}

async function quickBuy() {
  const code = document.getElementById('buy-code').value.trim();
  const shares = parseInt(document.getElementById('buy-shares').value) || 100;
  if (!code) { document.getElementById('trade-msg').textContent = '请输入股票代码'; return; }
  const msg = document.getElementById('trade-msg');
  msg.textContent = '交易中...';
  try {
    const r = await fetch('/api/trade/buy', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code,shares})});
    const d = await r.json();
    msg.textContent = d.message;
    msg.style.color = d.success ? 'var(--green)' : 'var(--red)';
    if (d.success) refreshAccount();
  } catch(e) { msg.textContent = '交易失败'; msg.style.color = 'var(--red)'; }
}

async function quickSell() {
  const code = document.getElementById('buy-code').value.trim();
  const shares = parseInt(document.getElementById('buy-shares').value) || 100;
  if (!code) { document.getElementById('trade-msg').textContent = '请输入股票代码'; return; }
  const msg = document.getElementById('trade-msg');
  msg.textContent = '交易中...';
  try {
    const r = await fetch('/api/trade/sell', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code,shares})});
    const d = await r.json();
    msg.textContent = d.message;
    msg.style.color = d.success ? 'var(--green)' : 'var(--red)';
    if (d.success) refreshAccount();
  } catch(e) { msg.textContent = '交易失败'; msg.style.color = 'var(--red)'; }
}

async function refreshAccount() {
  try {
    const r = await fetch('/api/trade/account');
    const d = await r.json();
    document.getElementById('acct-total').textContent = '¥'+d.total_value.toLocaleString();
    document.getElementById('acct-cash').textContent = '¥'+d.balance.toLocaleString();
    document.getElementById('acct-mktval').textContent = '¥'+(d.market_value||0).toLocaleString();
    const pnlEl = document.getElementById('acct-pnl');
    pnlEl.textContent = (d.total_pnl>=0?'+':'')+'¥'+d.total_pnl.toLocaleString();
    pnlEl.style.color = d.total_pnl>=0 ? 'var(--red)' : 'var(--green)';
    const todayEl = document.getElementById('acct-today-pnl');
    todayEl.textContent = ((d.today_pnl||0)>=0?'+':'')+'¥'+(d.today_pnl||0).toLocaleString();
    todayEl.style.color = (d.today_pnl||0)>=0 ? 'var(--red)' : 'var(--green)';

    const posDiv = document.getElementById('positions-list');
    if (!d.positions.length) {
      posDiv.innerHTML = '<span style="color:var(--text2)">暂无持仓</span><div id="positions-actions" style="margin-top:8px"></div>';
    } else {
      let h = '';
      d.positions.forEach(p => {
        const todayColor = (p.today_pnl||0) >= 0 ? 'var(--red)' : 'var(--green)';
        const pctColor = (p.pct_change||0) >= 0 ? 'var(--red)' : 'var(--green)';
        const totalColor = p.pnl >= 0 ? 'var(--red)' : 'var(--green)';
        const weightPct = d.total_value > 0 ? (p.market_value / d.total_value * 100).toFixed(1) : 0;
        const frozen = p.frozen ? ' (T+1冻结)' : '';
        h += '<div class="pos-row pos-clickable" style="flex-wrap:wrap;padding:10px" onclick="showDetailModal(\'' + p.code + '\')">'
        + '<div style="display:flex;align-items:center;width:100%;justify-content:space-between;margin-bottom:4px">'
          + '<span class="pos-code">' + p.code + '</span>'
          + '<span style="font-size:12px;color:var(--text)">' + (p.name||'') + '</span>'
          + '<span style="font-size:11px;color:var(--text2)">' + (p.industry||'') + '</span>'
          + '<span style="font-size:11px;color:var(--text2)">权重 ' + weightPct + '%' + frozen + '</span>'
        + '</div>'
        + '<div style="display:flex;align-items:center;width:100%;justify-content:space-between;font-size:12px">'
          + '<span class="pos-shares">' + p.shares + '股</span>'
          + '<span style="color:var(--text2);min-width:70px;text-align:right">成本 ¥' + p.avg_cost + '</span>'
          + '<span style="min-width:70px;text-align:right;font-weight:600">现价 ¥' + p.current_price + '</span>'
          + '<span style="min-width:70px;text-align:right;color:' + pctColor + ';font-weight:600">' + (p.pct_change>=0?'+':'') + p.pct_change + '%</span>'
        + '</div>'
        + '<div style="display:flex;align-items:center;width:100%;justify-content:space-between;font-size:12px;margin-top:2px">'
          + '<span style="color:var(--text2)">市值 ¥' + p.market_value.toLocaleString() + '</span>'
          + '<span style="color:' + todayColor + '">今日 ' + (p.today_pnl>=0?'+':'') + '¥' + p.today_pnl.toLocaleString() + '</span>'
          + '<span style="color:' + totalColor + ';font-weight:600">累计 ' + (p.pnl>=0?'+':'') + '¥' + p.pnl.toLocaleString() + ' (' + (p.pnl_pct>=0?'+':'') + p.pnl_pct + '%)</span>'
        + '</div>'
        + '<div style="width:100%;text-align:right;margin-top:4px" onclick="event.stopPropagation()">'
          + '<button class="btn-danger" onclick="sellPosition(\'' + p.code + '\',' + (p.shares - (p.frozen||0)) + ')" style="font-size:10px;padding:2px 8px">卖出</button>'
        + '</div></div>';
      });
      h += '<div id="positions-actions" style="margin-top:8px;text-align:right">'
        + '<button class="btn-danger" onclick="sellAllPositions()">一键卖出全部</button></div>';
      posDiv.innerHTML = h;
    }

    // 交易记录 (需求5: 增加股票名称)
    const txnDiv = document.getElementById('txn-list');
    const txns = await (await fetch('/api/trade/transactions')).json();
    if (!txns.length) { txnDiv.innerHTML = '<span style="color:var(--text2)">暂无交易记录</span>'; }
    else {
      txnDiv.innerHTML = txns.map(t =>
        '<div class="txn-row">'
        + '<span style="min-width:85px">'+t.date+'</span>'
        + '<span style="color:'+(t.action==='买入'?'var(--red)':'var(--green)')+';min-width:36px;font-weight:600">'+t.action+'</span>'
        + '<span style="color:var(--accent);min-width:68px">'+t.code+'</span>'
        + '<span style="font-size:11px;color:var(--text);min-width:55px">'+(t.name||'')+'</span>'
        + '<span style="font-size:10px;color:var(--text2)">'+t.shares+'股 @ ¥'+t.price+'</span>'
        + '<span style="margin-left:auto;color:'+(t.net>=0?'var(--green)':'var(--red)')+'">'+(t.net>=0?'+':'')+'¥'+t.net.toFixed(2)+'</span>'
        + '</div>'
      ).join('');
    }

    // 缓存快照数据供走势图弹窗使用
    const snaps = await (await fetch('/api/trade/snapshots')).json();
    window._pnlSnaps = snaps;
  } catch(e) {}
}

async function loadPnlCharts() {
  const snaps = window._pnlSnaps;
  if (!snaps || snaps.length < 2) return;
  const ChartBg = '#0d1117';
  const ChartGrid = '#21262d';
  const pnlValues = snaps.map(s => (s.total_value||0) - (s.total_recharged||s.total_value||0));
  const pnlData = [{x:snaps.map(s=>s.date), y:pnlValues, type:'scatter', mode:'lines+markers',
    line:{color:'#58a6ff',width:2}, fill:'tozeroy', fillcolor:'rgba(88,166,255,0.08)', name:'累计盈亏'}];
  const pnlLayout = {margin:{l:50,r:10,t:10,b:40},height:240,
    paper_bgcolor:ChartBg,plot_bgcolor:ChartBg,
    xaxis:{gridcolor:ChartGrid,color:'#8b949e'},
    yaxis:{gridcolor:ChartGrid,color:'#8b949e',tickprefix:'¥',zeroline:true,zerolinecolor:'#30363d',zerolinewidth:1},
    showlegend:false};
  Plotly.react('chart-pnl-amt', pnlData, pnlLayout, {responsive:true});

  const pnlPct = snaps.map(s => {
    const base = s.total_recharged || s.total_value || 1;
    return (s.total_value - base) / Math.max(base, 1) * 100;
  });
  const pnlPctData = [{x:snaps.map(s=>s.date), y:pnlPct, type:'scatter', mode:'lines+markers',
    line:{color:'#3fb950',width:2}, fill:'tozeroy', fillcolor:'rgba(63,185,80,0.08)', name:'收益率%'}];
  const pnlPctLayout = {margin:{l:50,r:10,t:10,b:40},height:240,
    paper_bgcolor:ChartBg,plot_bgcolor:ChartBg,
    xaxis:{gridcolor:ChartGrid,color:'#8b949e'},
    yaxis:{gridcolor:ChartGrid,color:'#8b949e',ticksuffix:'%',zeroline:true,zerolinecolor:'#30363d',zerolinewidth:1},
    showlegend:false};
  Plotly.react('chart-pnl-pct', pnlPctData, pnlPctLayout, {responsive:true});
}

function showPnlChartModal() {
  document.getElementById('pnl-chart-modal').style.display = 'flex';
  loadPnlCharts();
}
function hidePnlChartModal() {
  document.getElementById('pnl-chart-modal').style.display = 'none';
}


function showRechargeModal() {
  document.getElementById('recharge-modal').style.display = 'flex';
}
function hideRechargeModal() {
  document.getElementById('recharge-modal').style.display = 'none';
  document.getElementById('recharge-msg').textContent = '';
}
function quickRecharge(amount) {
  document.getElementById('recharge-amount').value = amount;
}
async function doRecharge() {
  const amt = parseFloat(document.getElementById('recharge-amount').value);
  if (!amt || amt <= 0) { document.getElementById('recharge-msg').textContent = '请输入有效金额'; document.getElementById('recharge-msg').style.color = 'var(--red)'; return; }
  try {
    const r = await fetch('/api/trade/recharge', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({amount:amt})});
    const d = await r.json();
    document.getElementById('recharge-msg').textContent = d.message;
    document.getElementById('recharge-msg').style.color = d.success ? 'var(--green)' : 'var(--red)';
    if (d.success) { refreshAccount(); setTimeout(hideRechargeModal, 800); }
  } catch(e) { document.getElementById('recharge-msg').textContent = '网络错误'; }
}

async function resetAccount() {
  if (!confirm('确定要清除所有虚拟账户数据吗？\n\n此操作不可恢复！\n将清空：持仓、交易记录、盈亏快照、资金余额')) return;
  try {
    const r = await fetch('/api/trade/reset', {method:'POST',headers:{'Content-Type':'application/json'}});
    const d = await r.json();
    if (d.success) {
      showToast('账户已重置');
      refreshAccount();
    } else {
      showToast('重置失败: ' + d.message);
    }
  } catch(e) { showToast('网络错误'); }
}

let origSwitchTab = switchTab;
switchTab = function(tab) {
  origSwitchTab(tab);
  if (tab === 'strategy') { refreshAccount(); loadCachedPicks(); }
};

initTheme();
init();
checkAuthState();
</script>
</body>
</html>"""


# ============================================================
# 认证装饰器
# ============================================================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "请先登录"}), 401
        # 验证用户是否仍存在于数据库
        from auth import get_user_by_id
        if get_user_by_id(session["user_id"]) is None:
            session.clear()
            return jsonify({"error": "用户不存在，请重新登录"}), 401
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "请先登录"}), 401
        if not session.get("is_admin"):
            return jsonify({"error": "需要管理员权限"}), 403
        # 验证管理员是否仍存在
        from auth import get_user_by_id
        user = get_user_by_id(session["user_id"])
        if user is None or not user.get("is_admin"):
            session.clear()
            return jsonify({"error": "权限已失效，请重新登录"}), 403
        return f(*args, **kwargs)
    return wrapper


def csrf_protect(f):
    """简易 CSRF 保护：要求 POST 请求带 application/json Content-Type。
    浏览器跨域 <form> 提交无法伪造此 Content-Type，可防止简单 CSRF 攻击。"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            ct = (request.content_type or "").lower()
            if "application/json" not in ct:
                return jsonify({"error": "请求格式无效"}), 400
        return f(*args, **kwargs)
    return wrapper


# ============================================================
# 认证 API
# ============================================================

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    ok, msg = register_user(data.get("username", ""), data.get("password", ""))
    return jsonify({"success": ok, "message": msg})


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    ok, result = auth_login(data.get("username", ""), data.get("password", ""))
    if ok:
        session["user_id"] = result["id"]
        session["username"] = result["username"]
        session["is_admin"] = result["is_admin"]
        return jsonify({"success": True, "user": result})
    return jsonify({"success": False, "message": result})


@app.route("/api/auth/logout", methods=["POST"])
@csrf_protect
def api_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me")
def api_me():
    if "user_id" not in session:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "user": {
            "id": session["user_id"],
            "username": session["username"],
            "is_admin": session.get("is_admin", False),
        },
    })


# ============================================================
# 自选股 API
# ============================================================

@app.route("/api/watchlist")
@login_required
def api_watchlist():
    wl = get_watchlist(session["user_id"])
    # 补充股票名称和最新信息
    result = []
    for item in wl:
        code = item["code"]
        name = NAMES.get(code, "")
        industry = INDUSTRIES.get(code, "")
        price = None
        pct = None
        if code in STOCKS:
            df = STOCKS[code]
            price = round(float(df["close"].iloc[-1]), 2)
            pct = round(float(df.iloc[-1].get("pct_change", 0) or 0), 2)
        result.append({
            "code": code,
            "name": name,
            "industry": industry,
            "latest": price,
            "pct_change": pct,
            "added_at": item["added_at"],
        })
    return jsonify(result)


@app.route("/api/watchlist/add", methods=["POST"])
@login_required
@csrf_protect
def api_watchlist_add():
    data = request.get_json() or {}
    ok, msg = add_to_watchlist(session["user_id"], data.get("code", ""))
    return jsonify({"success": ok, "message": msg})


@app.route("/api/watchlist/remove", methods=["POST"])
@login_required
@csrf_protect
def api_watchlist_remove():
    data = request.get_json() or {}
    ok, msg = remove_from_watchlist(session["user_id"], data.get("code", ""))
    return jsonify({"success": ok, "message": msg})


# ============================================================
# 管理员 API
# ============================================================

@app.route("/api/admin/users")
@admin_required
def api_admin_users():
    users = admin_list_users()
    return jsonify(users)


@app.route("/api/admin/add-user", methods=["POST"])
@admin_required
@csrf_protect
def api_admin_add_user():
    data = request.get_json() or {}
    ok, msg = admin_add_user(
        data.get("username", ""),
        data.get("password", ""),
        data.get("is_admin", False),
    )
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/delete-user", methods=["POST"])
@admin_required
@csrf_protect
def api_admin_delete_user():
    data = request.get_json() or {}
    ok, msg = admin_delete_user(session["user_id"], data.get("user_id", 0))
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/user-watchlist/<int:user_id>")
@admin_required
def api_admin_user_watchlist(user_id):
    wl = admin_get_user_watchlist(user_id)
    result = []
    for item in wl:
        code = item["code"]
        name = NAMES.get(code, "")
        result.append({
            "code": code,
            "name": name,
            "added_at": item["added_at"],
        })
    return jsonify(result)


@app.route("/")
def index():
    return render_template_string(
        HTML,
        stock_list_json=json.dumps(
            [{k: (None if isinstance(v, float) and np.isnan(v) else v)
              for k, v in r.items()}
             for r in SUMMARY_DF.to_dict("records")],
            ensure_ascii=False
        ).replace("</", "<\\/"),
        n_stocks=len(STOCKS),
        n_rows=f"{len(FULL):,}",
        n_industries=len(set(INDUSTRIES.values())),
        sync_message=sync_result["message"],
        sync_latest_trading=sync_result["latest_trading"],
    )


@app.route("/api/chart/<code>")
def chart_data(code: str):
    df = STOCKS.get(code)
    if df is None:
        return jsonify({"error": "not found"}), 404

    # -- 走势图 --
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], mode="lines",
        name=code, line=dict(color="#58a6ff", width=1.5),
    ))
    price_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=300,
    )

    # -- 收益率分布 --
    ret_clean = df["ret"].dropna() * 100
    ret_fig = go.Figure()
    ret_fig.add_trace(go.Histogram(
        x=ret_clean, nbinsx=80,
        marker=dict(color="#58a6ff", line=dict(color="#30363d", width=1)),
    ))
    ret_fig.add_vline(x=0, line_width=1, line_color="#f85149")
    ret_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=300,
        
        xaxis=dict(title="收益率 (%)"),
    )

    # -- 成交量 --
    colors = ["#3fb950" if df["close"].iloc[i] >= df["close"].iloc[i - 1]
              else "#f85149" for i in range(1, len(df))]
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(
        x=df["date"].iloc[1:], y=df["volume"].iloc[1:] / 1e4,
        marker=dict(color=colors), name="成交量(万手)",
    ))
    vol_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=220,
    )

    # -- 估值双轴图：换手率 + PE --
    val_fig = make_subplots(specs=[[{"secondary_y": True}]])
    val_fig.add_trace(go.Scatter(
        x=df["date"], y=df["pe_ttm"], mode="lines",
        name="PE(TTM)", line=dict(color="#f0883e", width=1),
    ), secondary_y=False)
    val_fig.add_trace(go.Scatter(
        x=df["date"], y=df["turnover"], mode="lines",
        name="换手率(%)", line=dict(color="#58a6ff", width=1, dash="dot"),
    ), secondary_y=True)
    val_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=220,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    val_fig.update_yaxes(title_text="PE", secondary_y=False)
    val_fig.update_yaxes(title_text="换手率 %", secondary_y=True)

    # 统计
    ret_total = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    sharpe = (df["ret"].mean() / df["ret"].std() * np.sqrt(252)) if df["ret"].std() > 0 else 0

    # -- 财务图表: ROE & 净利率趋势 --
    fin_df = FINANCIALS.get(code)
    roe_fig = go.Figure()
    dupont_fig = go.Figure()
    fin_stats = {}
    if fin_df is not None and not fin_df.empty:
        fdf = fin_df.sort_values("stat_date")
        roe_fig.add_trace(go.Scatter(
            x=fdf["stat_date"], y=fdf["roe_avg"], mode="lines+markers",
            name="ROE(%)", line=dict(color="#f0883e", width=2),
        ))
        roe_fig.add_trace(go.Scatter(
            x=fdf["stat_date"], y=fdf["np_margin"], mode="lines+markers",
            name="净利率(%)", line=dict(color="#58a6ff", width=2),
        ))
        # 杜邦分析
        dupont_fig.add_trace(go.Bar(
            x=fdf["stat_date"], y=fdf["dupont_roe"], name="ROE(%)",
            marker=dict(color="#f0883e"),
        ))
        dupont_fig.add_trace(go.Bar(
            x=fdf["stat_date"], y=fdf["dupont_asset_to_equity"], name="权益乘数",
            marker=dict(color="#58a6ff"),
        ))
        dupont_fig.add_trace(go.Bar(
            x=fdf["stat_date"], y=fdf["dupont_ni_to_gr"], name="净利率×总资产周转率",
            marker=dict(color="#3fb950"),
        ))
        latest_f = fdf.iloc[-1]
        fin_stats = {
            "roe": round(float(latest_f.get("roe_avg", 0) or 0), 1),
            "np_margin": round(float(latest_f.get("np_margin", 0) or 0), 1),
            "debt_ratio": round(float(latest_f.get("liability_to_asset", 0) or 0), 1),
            "yoy_ni": round(float(latest_f.get("yoy_ni", 0) or 0), 1),
        }
    roe_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    dupont_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode="group",
    )

    # 数据最新日期和同步时间
    last_data_date = str(df["date"].max())[:10]
    sync_times = get_stock_sync_time(code)
    if not sync_times:
        fpath = DATA_DIR / f"{code}.parquet"
        if fpath.exists():
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")
            sync_times = {"daily": mtime, "financial": ""}

    # ---- 详细交易数据 ----
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    tail5 = df.tail(5)

    # 涨跌停价格（根据板块判断倍率: 688=科创板, 300/301=创业板 → 20%）
    if code.startswith("688"):
        limit_mult = 1.20
    elif code.startswith(("300", "301")):
        limit_mult = 1.20
    else:
        limit_mult = 1.10
    pre_close = float(latest.get("pre_close", latest["close"]))
    limit_up = round(pre_close * limit_mult, 2)
    limit_down = round(pre_close * (2 - limit_mult), 2)

    # 量比 = 今日成交量 / 近5日均量
    avg_vol_5d = float(tail5["volume"].mean()) if len(tail5) >= 5 else float(latest["volume"])
    vol_ratio = round(float(latest["volume"]) / avg_vol_5d, 2) if avg_vol_5d > 0 else 1.0

    # 成交量格式化 (股 → 万手)
    vol_shou = float(latest["volume"]) / 100
    if vol_shou >= 10000:
        vol_str = f"{vol_shou/10000:.2f}万手"
    else:
        vol_str = f"{vol_shou:.0f}手"

    # 成交额格式化
    amt = float(latest["amount"])
    if amt >= 1e8:
        amt_str = f"{amt/1e8:.3f}亿"
    elif amt >= 1e4:
        amt_str = f"{amt/1e4:.2f}万"
    else:
        amt_str = f"{amt:.0f}元"

    # 市值 = 股价 × 总股本 / 流通股本（从最新财务数据）
    total_cap = None
    circ_cap = None
    fin_df = FINANCIALS.get(code)
    if fin_df is not None and not fin_df.empty:
        latest_fin = fin_df.sort_values("stat_date").iloc[-1]
        ts = float(latest_fin.get("total_share", 0) or 0)
        ls = float(latest_fin.get("liqa_share", 0) or 0)
        close_p = float(latest["close"])
        if ts > 0:
            total_cap = f"{close_p * ts / 1e8:.1f}亿"
        if ls > 0:
            circ_cap = f"{close_p * ls / 1e8:.1f}亿"

    # 市盈(动) 和 市净
    pe_val = round(float(latest["pe_ttm"]), 2) if "pe_ttm" in df.columns and not pd.isna(latest["pe_ttm"]) and float(latest["pe_ttm"]) > 0 else None
    pb_val = round(float(latest["pb_mrq"]), 2) if "pb_mrq" in df.columns and not pd.isna(latest["pb_mrq"]) else None

    base_stats = {
        "code": code,
        "name": NAMES.get(code, ""),
        "industry": INDUSTRIES.get(code, "未知"),
        "latest": round(float(latest["close"]), 2),
        "pct_change": round(float(latest.get("pct_change", 0) or 0), 2),
        # 上排
        "open": round(float(latest["open"]), 2),
        "high": round(float(latest["high"]), 2),
        "limit_up": limit_up,
        "turnover": round(float(latest.get("turnover", 0) or 0), 2),
        "volume_str": vol_str,
        "pe": pe_val,
        "total_cap": total_cap or "--",
        # 下排
        "pre_close": pre_close,
        "low": round(float(latest["low"]), 2),
        "limit_down": limit_down,
        "vol_ratio": vol_ratio,
        "amount_str": amt_str,
        "pb": pb_val,
        "circ_cap": circ_cap or "--",
        # 其他
        "return": round(ret_total, 1),
        "sharpe": round(sharpe, 2),
        "last_data_date": last_data_date,
        "last_daily_sync": sync_times.get("daily", ""),
        "last_fin_sync": sync_times.get("financial", ""),
    }
    base_stats.update(fin_stats)

    for fig in [price_fig, ret_fig, vol_fig, val_fig, roe_fig, dupont_fig]:
        fig.layout.template = None
    return jsonify({
        "price": json.loads(price_fig.to_json()),
        "ret_hist": json.loads(ret_fig.to_json()),
        "volume": json.loads(vol_fig.to_json()),
        "valuation": json.loads(val_fig.to_json()),
        "roe_chart": json.loads(roe_fig.to_json()),
        "dupont_chart": json.loads(dupont_fig.to_json()),
        "stats": base_stats,
    })


@app.route("/api/sync-status")
def sync_status():
    """返回后台同步的实时状态"""
    return jsonify(get_sync_status())


@app.route("/api/sync/trigger", methods=["POST"])
@admin_required
@csrf_protect
def api_sync_trigger():
    """手动触发同步"""
    result = trigger_manual_sync()
    return jsonify(result)


@app.route("/api/sync/historical", methods=["POST"])
@admin_required
@csrf_protect
def api_sync_historical():
    """拉取指定时间段的日线数据"""
    data = request.get_json() or {}
    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")
    if not start_date or not end_date:
        return jsonify({"ok": False, "message": "请指定起止日期"}), 400

    # 检查是否已在运行
    status = get_sync_status()
    if status.get("running"):
        return jsonify({"ok": False, "message": "同步已在运行中，请等待完成"})

    t = threading.Thread(target=_run_hist_sync, args=(start_date, end_date), daemon=True)
    t.start()
    return jsonify({"ok": True, "message": f"历史数据拉取已启动: {start_date} ~ {end_date}"})


@app.route("/api/data/summary")
def api_data_summary():
    """返回数据时间范围概览"""
    return jsonify(get_data_summary())


@app.route("/api/compare")
def compare():
    a, b = request.args.get("a", ""), request.args.get("b", "")
    fig = go.Figure()
    for code, color in [(a, "#58a6ff"), (b, "#f0883e")]:
        df = STOCKS.get(code)
        if df is None:
            continue
        norm = df["close"] / df["close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df["date"], y=norm, mode="lines",
            name=f"{code} {NAMES.get(code,'')}", line=dict(color=color, width=1.5),
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=300,
        yaxis=dict(title="归一化 (Base=100)"),
    )
    fig.layout.template = None
    return jsonify({"price": json.loads(fig.to_json())})


# ============================================================
# 纸交易 API
# ============================================================

_prices_cache: dict = {}
_prices_cache_time: float = 0

def _get_current_prices():
    """从已加载的STOCKS获取最新收盘价（缓存60秒，过期自动刷新）"""
    global _prices_cache, _prices_cache_time
    import time as _time
    now = _time.time()
    if _prices_cache and (now - _prices_cache_time) < 60:
        return _prices_cache
    prices = {}
    for code, df in STOCKS.items():
        if len(df) > 0:
            prices[code] = float(df["close"].iloc[-1])
    _prices_cache = prices
    _prices_cache_time = now
    return prices


@app.route("/api/market/summary")
def api_market_summary():
    up = sum(1 for s in SUMMARY if (s.get("pct_change") or 0) > 0)
    down = sum(1 for s in SUMMARY if (s.get("pct_change") or 0) < 0)
    flat_count = len(SUMMARY) - up - down
    avg_pct = sum(s.get("pct_change") or 0 for s in SUMMARY) / max(len(SUMMARY), 1)
    # 行业板块表现
    ind_pct = {}
    ind_count = {}
    for s in SUMMARY:
        ind = s.get("industry") or "未知"
        pct = s.get("pct_change") or 0
        ind_pct[ind] = ind_pct.get(ind, 0) + pct
        ind_count[ind] = ind_count.get(ind, 0) + 1
    ind_avg = {k: round(v / max(ind_count[k], 1), 2) for k, v in ind_pct.items()}
    top_sectors = sorted(ind_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    bottom_sectors = sorted(ind_avg.items(), key=lambda x: x[1])[:5]
    # 总成交额和平均换手率
    total_amount = 0.0
    total_turnover = 0.0
    turnover_count = 0
    pe_values = []
    for code, df in STOCKS.items():
        if len(df) > 0:
            latest = df.iloc[-1]
            total_amount += float(latest.get("amount", 0) or 0)
            t = latest.get("turnover")
            if t is not None and not pd.isna(t):
                total_turnover += float(t)
                turnover_count += 1
            pe = latest.get("pe_ttm")
            if pe is not None and not pd.isna(pe) and float(pe) > 0:
                pe_values.append(float(pe))
    avg_pe = round(sum(pe_values) / max(len(pe_values), 1), 1) if pe_values else None
    avg_turnover = round(total_turnover / max(turnover_count, 1), 2)
    return jsonify({
        "up": up, "down": down, "flat": flat_count,
        "avg_pct": round(avg_pct, 2), "total": len(SUMMARY),
        "top_sectors": [{"name": n, "pct": p} for n, p in top_sectors],
        "bottom_sectors": [{"name": n, "pct": p} for n, p in bottom_sectors],
        "total_amount": round(total_amount, 0),
        "avg_turnover": avg_turnover,
        "avg_pe": avg_pe,
    })

@app.route("/api/strategy/recommend/start", methods=["POST"])
@login_required
@csrf_protect
def api_strategy_recommend_start():
    """异步启动推荐计算，通过 /api/strategy/recommend/status 跟踪进度"""
    # 检查后台同步是否正在运行（同步期间读取 parquet 会冲突）
    sync_status = get_sync_status()
    if sync_status.get("running"):
        return jsonify({"success": False, "message": f"数据同步进行中（{sync_status.get('phase','')}）请等待同步完成后重试"})
    engine = get_engine(session.get("user_id", 0))
    result = engine.get_recommendations_async(top_n=10)
    return jsonify(result)


@app.route("/api/strategy/recommend/cached")
@login_required
def api_strategy_recommend_cached():
    """返回缓存的今日推荐（无缓存则返回空）"""
    cached = _load_cached_recommendation()
    if cached:
        return jsonify(cached)
    return jsonify({"success": False, "message": "暂无今日推荐"})


@app.route("/api/strategy/recommend/status")
@login_required
def api_strategy_recommend_status():
    """返回推荐计算进度"""
    return jsonify(get_rec_status())


@app.route("/api/strategy/recommend")
@login_required
def api_strategy_recommend():
    """同步获取推荐"""
    rec_status = get_rec_status()
    if rec_status["running"]:
        return jsonify({"success": False, "message": "推荐计算进行中，请等待..."})
    if rec_status.get("result"):
        try:
            return jsonify(rec_status["result"])
        except Exception as e:
            print(f"[dashboard] 缓存推荐结果序列化失败: {e}，重新计算...")
    try:
        engine = get_engine(session.get("user_id", 0))
        result = engine.get_recommendations(top_n=10)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"策略引擎错误: {e}"})


@app.route("/api/trade/account")
@login_required
def api_trade_account():
    engine = get_engine(session.get("user_id", 0))
    prices = _get_current_prices()
    # 获取昨日收盘价用于计算今日盈亏
    prev_prices = {}
    for code, df in STOCKS.items():
        if len(df) >= 2:
            prev_prices[code] = float(df["close"].iloc[-2])
        elif len(df) > 0:
            prev_prices[code] = float(df["close"].iloc[-1])
    # 先记录快照
    today = datetime.now().strftime("%Y-%m-%d")
    engine.take_snapshot(prices, today)
    summary = engine.get_account_summary(prices, prev_prices)
    # 补充股票名称
    for pos in summary["positions"]:
        pos["name"] = NAMES.get(pos["code"], "")
        pos["industry"] = INDUSTRIES.get(pos["code"], "未知")
    return jsonify(summary)


@app.route("/api/trade/recharge", methods=["POST"])
@login_required
@csrf_protect
def api_trade_recharge():
    data = request.get_json() or {}
    try:
        amount = float(data.get("amount", 0))
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "金额格式无效"}), 400
    engine = get_engine(session.get("user_id", 0))
    result = engine.recharge(amount)
    return jsonify(result)


@app.route("/api/trade/buy", methods=["POST"])
@login_required
@csrf_protect
def api_trade_buy():
    data = request.get_json() or {}
    code = (data.get("code") or "").strip()
    if not code:
        return jsonify({"success": False, "message": "股票代码不能为空"}), 400
    try:
        shares = int(data.get("shares", 100))
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "股数格式无效"}), 400
    prices = _get_current_prices()
    price = prices.get(code)
    if price is None:
        return jsonify({"success": False, "message": f"找不到 {code} 的价格"})
    trade_date = datetime.now().strftime("%Y-%m-%d")
    engine = get_engine(session.get("user_id", 0))
    result = engine.buy(code, shares, price, trade_date)
    return jsonify(result)


@app.route("/api/trade/sell", methods=["POST"])
@login_required
@csrf_protect
def api_trade_sell():
    data = request.get_json() or {}
    code = (data.get("code") or "").strip()
    if not code:
        return jsonify({"success": False, "message": "股票代码不能为空"}), 400
    try:
        shares = int(data.get("shares", 100))
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "股数格式无效"}), 400
    prices = _get_current_prices()
    price = prices.get(code)
    if price is None:
        return jsonify({"success": False, "message": f"找不到 {code} 的价格"})
    trade_date = datetime.now().strftime("%Y-%m-%d")
    engine = get_engine(session.get("user_id", 0))
    result = engine.sell(code, shares, price, trade_date)
    return jsonify(result)


@app.route("/api/trade/snapshots")
@login_required
def api_trade_snapshots():
    engine = get_engine(session.get("user_id", 0))
    snaps = engine.get_daily_snapshots(90)
    total_recharged = engine.account.get("total_recharged", 0)
    for s in snaps:
        s["total_recharged"] = total_recharged
    return jsonify(snaps)


@app.route("/api/trade/transactions")
@login_required
def api_trade_transactions():
    engine = get_engine(session.get("user_id", 0))
    txns = engine.get_transactions(50)
    # 补充股票名称
    for t in txns:
        t["name"] = NAMES.get(t.get("code", ""), "")
    return jsonify(txns)


@app.route("/api/trade/sell_all", methods=["POST"])
@login_required
@csrf_protect
def api_trade_sell_all():
    """一键卖出所有持仓"""
    engine = get_engine(session.get("user_id", 0))
    prices = _get_current_prices()
    trade_date = datetime.now().strftime("%Y-%m-%d")
    results = []
    for code in list(engine.account.get("positions", {}).keys()):
        pos = engine.account["positions"][code]
        price = prices.get(code, pos["avg_cost"])
        frozen = engine.account.get("frozen_shares", {}).get(code, 0)
        available = pos["shares"] - frozen
        if available >= 100:
            r = engine.sell(code, available, price, trade_date)
            results.append({"code": code, "shares": available, "success": r.get("success"), "message": r.get("message")})
    return jsonify({"success": True, "results": results, "count": len(results)})


@app.route("/api/trade/buy_basket", methods=["POST"])
@login_required
@csrf_protect
def api_trade_buy_basket():
    """一键按配比买入推荐股票"""
    data = request.get_json() or {}
    picks = data.get("picks", [])
    total_cash = float(data.get("total_cash", 0))
    if not picks or total_cash <= 0:
        return jsonify({"success": False, "message": "缺少推荐列表或资金不足"})

    prices = _get_current_prices()
    engine = get_engine(session.get("user_id", 0))
    # 使用账户全部可用资金（如果未指定）
    if total_cash > engine.account["balance"]:
        total_cash = engine.account["balance"]
    min_per_stock = 5000  # 每只股票至少5000元
    n = len(picks)
    results = []

    # 按权重分配资金
    weights = [p.get("weight", 1.0/n) for p in picks]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # 先计算每只股票的预算
    budgets = []
    for i, p in enumerate(picks):
        code = p["code"]
        price = prices.get(code)
        if not price or price <= 0:
            results.append({"code": code, "success": False, "message": "无价格数据"})
            budgets.append(0)
            continue
        budget = total_cash * weights[i]
        shares = int(budget / (price * 1.001) / 100) * 100
        if shares >= 100:
            budgets.append(shares * price * 1.001)
        else:
            budgets.append(0)

    # 调整：确保总额不超过可用资金，满足最低100股要求
    actual_total = sum(b for b in budgets if b > 0)
    if actual_total > engine.account["balance"]:
        scale = engine.account["balance"] / max(actual_total, 1)
        budgets = [b * scale for b in budgets]

    trade_date = datetime.now().strftime("%Y-%m-%d")
    for i, p in enumerate(picks):
        code = p["code"]
        price = prices.get(code)
        if not price or price <= 0:
            continue
        if budgets[i] < min_per_stock:
            continue
        shares = int(budgets[i] / (price * 1.001) / 100) * 100
        if shares < 100:
            continue
        actual_cost = shares * price * 1.001
        if actual_cost > engine.account["balance"]:
            shares = int(engine.account["balance"] / (price * 1.001) / 100) * 100
            if shares < 100:
                continue
        r = engine.buy(code, shares, price, trade_date)
        results.append({"code": code, "name": p.get("name", ""), "shares": shares,
                        "success": r.get("success"), "message": r.get("message")})

    return jsonify({"success": True, "results": results, "bought": sum(1 for r in results if r["success"])})


@app.route("/api/trade/reset", methods=["POST"])
@login_required
@csrf_protect
def api_trade_reset():
    """一键重置虚拟账户"""
    engine = get_engine(session.get("user_id", 0))
    data_file = engine.data_file
    engine.account = engine._default_account()
    engine.save()
    return jsonify({"success": True, "message": "账户已重置"})


@app.route("/api/stock/detail/<code>")
def api_stock_detail(code):
    """获取单只股票详细信息"""
    from collections import Counter
    if code not in STOCKS:
        return jsonify({"success": False, "message": "未找到该股票"})
    df = STOCKS[code]
    latest = df.iloc[-1]
    info = {
        "code": code,
        "name": NAMES.get(code, ""),
        "industry": INDUSTRIES.get(code, "未知"),
        "close": round(float(latest["close"]), 2),
        "pct_change": round(float(latest.get("pct_change", 0) or 0), 2),
        "volume": round(float(latest.get("volume", 0) or 0) / 1e6, 2),
        "amount": round(float(latest.get("amount", 0) or 0) / 1e8, 2),
        "pe_ttm": round(float(latest.get("pe_ttm", 0)), 1) if not pd.isna(latest.get("pe_ttm")) else None,
        "pb_mrq": round(float(latest.get("pb_mrq", 0)), 1) if not pd.isna(latest.get("pb_mrq")) else None,
        "turnover": round(float(latest.get("turnover", 0) or 0), 2),
        "last_date": str(df["date"].max())[:10],
        "ret_1m": round(float(df["close"].iloc[-1] / max(df["close"].iloc[-22] if len(df) >= 22 else df["close"].iloc[0], 1) - 1) * 100, 1) if len(df) >= 5 else None,
        "ret_3m": round(float(df["close"].iloc[-1] / max(df["close"].iloc[-66] if len(df) >= 66 else df["close"].iloc[0], 1) - 1) * 100, 1) if len(df) >= 15 else None,
        "high_52w": round(float(df["close"].tail(252).max()), 2) if len(df) >= 20 else None,
        "low_52w": round(float(df["close"].tail(252).min()), 2) if len(df) >= 20 else None,
    }
    # 财务数据
    if code in FINANCIALS:
        fin_df = FINANCIALS[code]
        latest_fin = fin_df.sort_values("stat_date").iloc[-1] if len(fin_df) > 0 else None
        if latest_fin is not None:
            for col in ["roe_avg", "np_margin", "gp_margin", "yoy_ni", "yoy_equity",
                        "current_ratio", "liability_to_asset", "asset_turn_ratio"]:
                if col in latest_fin.index and not pd.isna(latest_fin[col]):
                    info[col] = round(float(latest_fin[col]), 2)
    return jsonify({"success": True, "stock": info})


if __name__ == "__main__":
    print("\n  仪表盘: http://localhost:8050\n")
    app.run(host="0.0.0.0", port=8050, debug=False)
