"""
单股深度分析报告 — 东方财富风格

功能: K线+均线、PE/PB/PS/PCF估值走势、财务指标、收益分布
数据: 日线行情 + 财务报表 + 行业分类

用法: python report.py → 浏览器打开 http://localhost:8051
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "scripts"))
from data_fetcher import (
    load_names, load_industries, add_ma, add_atr, add_returns,
    load_all_financials,
)
import config as cfg

DAILY_DIR = ROOT / "data" / "daily"
FINANCIAL_DIR = ROOT / "data" / "financial"

# ============================================================
# 数据加载
# ============================================================
print("正在加载数据...")

STOCKS = {}
for f in sorted(DAILY_DIR.glob("*.parquet")):
    code = f.stem
    df = pd.read_parquet(f)
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = add_returns(df)
    df = add_ma(df)
    STOCKS[code] = df

NAMES = load_names().set_index("code")["name"].to_dict()
INDUSTRIES = load_industries().set_index("code")["industry"].to_dict()

# 加载财务数据
FINANCIALS = {}
if any(FINANCIAL_DIR.glob("*.parquet")):
    for f in sorted(FINANCIAL_DIR.glob("*.parquet")):
        code = f.stem
        df = pd.read_parquet(f)
        df["stat_date"] = pd.to_datetime(df["stat_date"])
        FINANCIALS[code] = df.sort_values("stat_date")
    print(f"财务数据: {len(FINANCIALS)} 只")
else:
    print("暂无财务数据")

CODES = sorted(STOCKS.keys())

# 构建摘要
SUMMARY = []
for code, df in STOCKS.items():
    if len(df) < 2:
        continue
    ret_total = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    sharpe = (df["ret"].mean() / df["ret"].std() * np.sqrt(252)) if df["ret"].std() > 0 else 0
    SUMMARY.append({
        "code": code,
        "name": NAMES.get(code, ""),
        "industry": INDUSTRIES.get(code, "未知"),
        "latest": round(df["close"].iloc[-1], 2),
        "return": round(ret_total, 1),
        "sharpe": round(sharpe, 2),
        "pe": round(df["pe_ttm"].iloc[-1], 1) if "pe_ttm" in df.columns and not pd.isna(df["pe_ttm"].iloc[-1]) else None,
        "pb": round(df["pb_mrq"].iloc[-1], 1) if "pb_mrq" in df.columns and not pd.isna(df["pb_mrq"].iloc[-1]) else None,
        "turnover": round(df["turnover"].mean(), 2) if "turnover" in df.columns else 0,
    })
SUMMARY_DF = pd.DataFrame(SUMMARY).sort_values("code")

print(f"加载完毕: {len(STOCKS)} 只股票, {len(FINANCIALS)} 只有财务数据, {len(INDUSTRIES)} 个行业")


# ============================================================
# HTML
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>单股深度分析 — 东方财富风格</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Microsoft YaHei',sans-serif;background:#f5f6fa;color:#333}
.header{background:#fff;border-bottom:2px solid #e53935;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 4px rgba(0,0,0,0.06)}
.header h1{font-size:18px;color:#c62828}
.header .stats{font-size:13px;color:#999}
.container{display:flex;height:calc(100vh - 49px)}
.sidebar{width:300px;background:#fff;border-right:1px solid #e8e8e8;overflow-y:auto;display:flex;flex-direction:column;box-shadow:2px 0 4px rgba(0,0,0,0.03)}
.sidebar input{width:100%;padding:10px 14px;border:none;border-bottom:1px solid #e8e8e8;font-size:14px;outline:none}
.sidebar input:focus{border-bottom-color:#e53935}
#stock-list{flex:1;overflow-y:auto}
.stock-item{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;cursor:pointer;font-size:12px;border-left:3px solid transparent;border-bottom:1px solid #f5f5f5;transition:background 0.15s}
.stock-item:hover{background:#fafafa}
.stock-item.active{background:#fff3f2;border-left-color:#e53935}
.stock-item .info{display:flex;flex-direction:column;gap:2px}
.stock-item .code{font-weight:600;font-size:13px;color:#333}
.stock-item .name{color:#999;font-size:11px}
.stock-item .right{text-align:right;display:flex;flex-direction:column;gap:2px}
.stock-item .ret{font-weight:600;font-size:12px}
.ret.pos{color:#e53935}.ret.neg{color:#1ca01c}
.stock-item .pe{color:#999;font-size:10px}
.main{flex:1;overflow-y:auto;padding:16px 20px}
/* 股票头部信息 */
.stock-header{background:linear-gradient(135deg,#c62828,#e53935);color:#fff;border-radius:8px;padding:16px 20px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center}
.stock-header .left{display:flex;align-items:baseline;gap:12px}
.stock-header .stock-name{font-size:20px;font-weight:700}
.stock-header .stock-code{font-size:12px;opacity:0.8}
.stock-header .stock-industry{font-size:11px;opacity:0.7;background:rgba(255,255,255,0.15);padding:2px 8px;border-radius:3px}
.stock-header .price-area{text-align:right}
.stock-header .latest-price{font-size:28px;font-weight:700;line-height:1}
.stock-header .price-meta{font-size:11px;opacity:0.8;margin-top:4px}
/* 指标卡片 */
.metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.metric-card{background:#fff;border-radius:8px;padding:14px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.metric-card .label{font-size:11px;color:#999;margin-bottom:6px}
.metric-card .value{font-size:20px;font-weight:700;color:#333}
.metric-card .sub{font-size:10px;color:#999;margin-top:4px}
.metric-card.highlight{border-left:3px solid #e53935}
/* 图表面板 */
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.chart-panel{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.chart-panel.full{grid-column:1/-1}
.chart-panel h3{font-size:13px;color:#333;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #f0f0f0;display:flex;align-items:center;gap:6px}
.chart-panel h3::before{content:'';display:inline-block;width:3px;height:14px;background:#e53935;border-radius:2px}
/* 表格 */
.fin-table{width:100%;font-size:11px;border-collapse:collapse;margin-top:6px}
.fin-table th{background:#fafafa;padding:6px 8px;border:1px solid #eee;text-align:center;font-weight:500;color:#666;white-space:nowrap}
.fin-table td{padding:5px 8px;border:1px solid #eee;text-align:center;white-space:nowrap}
.fin-table .col-label{text-align:left;font-weight:500;color:#333;background:#fafafa}
.fin-table .val-pos{color:#e53935}.fin-table .val-neg{color:#1ca01c}
#toast{position:fixed;top:16px;right:16px;background:#333;color:#fff;padding:10px 20px;border-radius:6px;display:none;z-index:1000;font-size:13px}
</style>
</head>
<body>
<div class="header">
  <h1>单股深度分析</h1>
  <div class="stats" id="header-stats"></div>
</div>
<div class="container">
  <div class="sidebar">
    <input type="text" id="search" placeholder="搜索代码或名称..." oninput="filterStocks()">
    <div id="stock-list"></div>
  </div>
  <div class="main" id="main-content">
    <div class="stock-header" id="stock-header"></div>
    <div class="metrics-row" id="metrics-row"></div>
    <div class="chart-row">
      <div class="chart-panel full"><h3>K线走势 & 均线</h3><div id="chart-kline" style="height:420px"></div></div>
    </div>
    <div class="chart-row">
      <div class="chart-panel"><h3>PE / PB 估值走势</h3><div id="chart-val" style="height:260px"></div></div>
      <div class="chart-panel"><h3>日收益率分布</h3><div id="chart-ret" style="height:260px"></div></div>
    </div>
    <div class="chart-row">
      <div class="chart-panel"><h3>ROE & 利润率趋势</h3><div id="chart-roe" style="height:260px"></div></div>
      <div class="chart-panel"><h3>成长性指标 (同比)</h3><div id="chart-growth" style="height:260px"></div></div>
    </div>
    <div class="chart-panel" style="margin-bottom:10px">
      <h3>财务指标明细</h3><div id="fin-table"></div>
    </div>
  </div>
</div>
<div id="toast"></div>
<script>
let allStocks = {{ stock_list_json | safe }};
let currentCode = '600519';

function init() {
  document.getElementById('header-stats').textContent =
    '{{ n_stocks }} 只 | {{ n_financials }} 只有财报 | {{ n_industries }} 个行业';
  renderStockList(allStocks);
  selectStock('600519');
}

function renderStockList(stocks) {
  document.getElementById('stock-list').innerHTML = stocks.map(s => `
    <div class="stock-item ${s.code===currentCode?'active':''}"
         onclick="selectStock('${s.code}')">
      <div class="info">
        <span class="code">${s.code} <span style="font-weight:400;font-size:11px;color:#999">${s.industry||''}</span></span>
        <span class="name">${s.name||''}</span>
      </div>
      <div class="right">
        <span class="ret ${s.return>=0?'pos':'neg'}">${s.return>=0?'+':''}${s.return}%</span>
        <span class="pe">PE ${s.pe||'--'} | PB ${s.pb||'--'}</span>
      </div>
    </div>
  `).join('');
}

function filterStocks() {
  const q = document.getElementById('search').value.toLowerCase();
  renderStockList(allStocks.filter(s => (s.code+s.name+s.industry).toLowerCase().includes(q)));
}

async function selectStock(code) {
  currentCode = code;
  renderStockList(allStocks);
  document.getElementById('search').value = '';
  try {
    const resp = await fetch('/api/report/'+code);
    const data = await resp.json();
    if(data.error){alert(data.error);return;}
    renderHeader(data.header);
    renderMetrics(data.metrics);
    Plotly.react('chart-kline', data.kline, {responsive:true});
    Plotly.react('chart-val', data.valuation, {responsive:true});
    Plotly.react('chart-ret', data.ret_hist, {responsive:true});
    Plotly.react('chart-roe', data.roe_chart, {responsive:true});
    Plotly.react('chart-growth', data.growth_chart, {responsive:true});
    renderFinTable(data.fin_table);
  } catch(e) {
    console.error(e);
    showToast('加载失败: '+e.message);
  }
}

function renderHeader(h) {
  const chgClass = h.pct_change>=0?'val-pos':'val-neg';
  const chgSign = h.pct_change>=0?'+':'';
  document.getElementById('stock-header').innerHTML = `
    <div class="left">
      <span class="stock-name">${h.name||h.code}</span>
      <span class="stock-code">${h.code}</span>
      <span class="stock-industry">${h.industry||'未知'}</span>
    </div>
    <div class="price-area">
      <div class="latest-price">¥${h.latest}</div>
      <div class="price-meta">
        涨跌幅 <span class="${chgClass}">${chgSign}${h.pct_change}%</span> &nbsp;
        成交 ${h.amount}亿 &nbsp; 换手 ${h.turnover}%
      </div>
    </div>`;
}

function renderMetrics(m) {
  const cards = [
    {label:'PE (TTM)', value:m.pe||'--', sub:'市盈率', hl:true},
    {label:'PB (MRQ)', value:m.pb||'--', sub:'市净率', hl:true},
    {label:'PS (TTM)', value:m.ps||'--', sub:'市销率', hl:false},
    {label:'PCF (TTM)', value:m.pcf||'--', sub:'市现率', hl:false},
    {label:'ROE', value:m.roe||'--', sub:'净资产收益率 %', hl:true},
    {label:'日均成交', value:(m.avg_amount||0)+'亿', sub:'日均成交额', hl:false},
    {label:'区间收益', value:(m.ret>=0?'+':'')+m.ret+'%', sub:'夏普 '+m.sharpe, hl:false},
    {label:'日均换手', value:(m.avg_turn||0)+'%', sub:'流动性指标', hl:false},
  ];
  document.getElementById('metrics-row').innerHTML = cards.map(c =>
    `<div class="metric-card${c.hl?' highlight':''}">
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      <div class="sub">${c.sub}</div>
    </div>`
  ).join('');
}

function renderFinTable(tb) {
  if(!tb || !tb.headers){
    document.getElementById('fin-table').innerHTML = '<div style="padding:20px;color:#999;text-align:center">暂无财务数据</div>';
    return;
  }
  let html = '<table class="fin-table"><thead><tr><th>指标</th>';
  tb.headers.forEach(h => html += `<th>${h}</th>`);
  html += '</tr></thead><tbody>';
  tb.rows.forEach(row => {
    html += '<tr><td class="col-label">'+row[0]+'</td>';
    for(let i=1;i<row.length;i++){
      html += '<td>'+row[i]+'</td>';
    }
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('fin-table').innerHTML = html;
}

function showToast(m) {
  const t=document.getElementById('toast');t.textContent=m;t.style.display='block';
  setTimeout(()=>t.style.display='none',2000);
}

init();
</script>
</body>
</html>"""


# ============================================================
# API
# ============================================================

@app.route("/")
def index():
    return render_template_string(
        HTML,
        stock_list_json=json.dumps(SUMMARY_DF.to_dict("records")),
        n_stocks=len(STOCKS),
        n_financials=len(FINANCIALS),
        n_industries=len(set(INDUSTRIES.values())),
    )


def _build_kline(df: pd.DataFrame, code: str) -> dict:
    """K线 + 均线 + 成交量"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.03,
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="K线",
        increasing=dict(line=dict(color="#e53935"), fillcolor="#e53935"),
        decreasing=dict(line=dict(color="#1ca01c"), fillcolor="#1ca01c"),
    ), row=1, col=1)

    # 均线
    ma_colors = {"ma_5": "#ff9800", "ma_10": "#2196f3", "ma_20": "#9c27b0", "ma_60": "#607d8b"}
    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[ma], mode="lines",
                name=ma.upper(), line=dict(color=color, width=1),
            ), row=1, col=1)

    # 成交量
    colors = ["#e53935" if df["close"].iloc[i] >= df["close"].iloc[i - 1]
              else "#1ca01c" for i in range(1, len(df))]
    fig.add_trace(go.Bar(
        x=df["date"].iloc[1:], y=df["volume"].iloc[1:] / 1e4,
        marker=dict(color=colors), name="成交量(万手)",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        xaxis2=dict(gridcolor="#f0f0f0"), yaxis2=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="top", y=1.12, x=0),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="万手", row=2, col=1)

    return json.loads(fig.to_json())


def _build_valuation(df: pd.DataFrame) -> dict:
    """PE / PB 双轴图"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["pe_ttm"], mode="lines",
        name="PE(TTM)", line=dict(color="#e53935", width=1.5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["pb_mrq"], mode="lines",
        name="PB(MRQ)", line=dict(color="#2196f3", width=1.5),
    ), secondary_y=True)

    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        xaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="PE", secondary_y=False, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="PB", secondary_y=True, gridcolor="#f0f0f0")

    return json.loads(fig.to_json())


def _build_ret_hist(df: pd.DataFrame) -> dict:
    """日收益率分布"""
    ret_clean = df["ret"].dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ret_clean, nbinsx=80,
        marker=dict(color="#e53935", line=dict(color="#fff", width=1)),
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#333", line_dash="dash")
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        xaxis=dict(title="日收益率 (%)", gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    return json.loads(fig.to_json())


def _build_roe_chart(fin_df: pd.DataFrame) -> dict:
    """ROE & 利润率趋势"""
    if fin_df is None or fin_df.empty:
        return {}

    fig = go.Figure()
    for col, name, color in [
        ("roe_avg", "ROE", "#e53935"),
        ("np_margin", "净利润率", "#2196f3"),
        ("gp_margin", "毛利率", "#ff9800"),
    ]:
        if col in fin_df.columns:
            valid = fin_df[fin_df[col].notna() & (fin_df[col] != 0)]
            fig.add_trace(go.Scatter(
                x=valid["stat_date"], y=valid[col] * 100, mode="lines+markers",
                name=name, line=dict(color=color, width=2),
                marker=dict(size=4),
            ))

    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(title="%", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return json.loads(fig.to_json())


def _build_growth_chart(fin_df: pd.DataFrame) -> dict:
    """成长性指标同比"""
    if fin_df is None or fin_df.empty:
        return {}

    fig = go.Figure()
    for col, name, color in [
        ("yoy_ni", "净利润同比", "#e53935"),
        ("yoy_equity", "净资产同比", "#2196f3"),
        ("yoy_asset", "总资产同比", "#ff9800"),
    ]:
        if col in fin_df.columns:
            valid = fin_df[fin_df[col].notna() & (fin_df[col] != 0)]
            fig.add_trace(go.Scatter(
                x=valid["stat_date"], y=valid[col] * 100, mode="lines+markers",
                name=name, line=dict(color=color, width=2),
                marker=dict(size=4),
            ))

    fig.add_hline(y=0, line_width=1, line_color="#ccc", line_dash="dash")
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(title="%", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return json.loads(fig.to_json())


def _build_fin_table(code: str) -> dict | None:
    """构建财务指标明细表"""
    fin = FINANCIALS.get(code)
    if fin is None or fin.empty:
        return None

    recent = fin.tail(8).copy()  # 最近8个季度
    headers = [str(d.date()) for d in recent["stat_date"]]

    indicators = [
        ("roe_avg", "ROE (%)"),
        ("np_margin", "净利率 (%)"),
        ("gp_margin", "毛利率 (%)"),
        ("eps_ttm", "EPS (TTM)"),
        ("current_ratio", "流动比率"),
        ("quick_ratio", "速动比率"),
        ("liability_to_asset", "资产负债率 (%)"),
        ("yoy_ni", "净利同比 (%)"),
        ("yoy_equity", "净资产同比 (%)"),
        ("cfo_to_np", "经营现金流/净利"),
        ("dupont_roe", "杜邦 ROE (%)"),
        ("asset_turn_ratio", "总资产周转率"),
        ("nr_turn_ratio", "应收周转率"),
        ("inv_turn_ratio", "存货周转率"),
    ]

    rows = []
    for col, label in indicators:
        if col not in recent.columns:
            continue
        vals = recent[col].tolist()
        row = [label]
        for v in vals:
            if pd.isna(v) or v == 0:
                row.append("--")
            elif col in ("roe_avg", "np_margin", "gp_margin", "yoy_ni",
                         "yoy_equity", "dupont_roe", "liability_to_asset"):
                row.append(f"{v * 100:.1f}")
            elif col == "eps_ttm":
                row.append(f"{v:.2f}")
            elif col in ("current_ratio", "quick_ratio", "cfo_to_np",
                         "asset_turn_ratio", "nr_turn_ratio", "inv_turn_ratio"):
                row.append(f"{v:.2f}")
            else:
                row.append(f"{v:.1f}")
        rows.append(row)

    return {"headers": headers, "rows": rows}


@app.route("/api/report/<code>")
def report_data(code: str):
    df = STOCKS.get(code)
    if df is None:
        return jsonify({"error": "not found"}), 404

    fin = FINANCIALS.get(code)

    # 头部信息
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pct_change = round((latest["close"] / prev["close"] - 1) * 100, 2)

    header = {
        "code": code,
        "name": NAMES.get(code, ""),
        "industry": INDUSTRIES.get(code, "未知"),
        "latest": round(latest["close"], 2),
        "pct_change": pct_change,
        "amount": round(latest["amount"] / 1e8, 1) if "amount" in latest else 0,
        "turnover": round(latest["turnover"], 2) if "turnover" in latest else 0,
    }

    # 指标卡片
    ret_total = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    sharpe = (df["ret"].mean() / df["ret"].std() * np.sqrt(252)) if df["ret"].std() > 0 else 0

    metrics = {
        "pe": round(latest["pe_ttm"], 1) if "pe_ttm" in latest and not pd.isna(latest.get("pe_ttm")) else None,
        "pb": round(latest["pb_mrq"], 1) if "pb_mrq" in latest and not pd.isna(latest.get("pb_mrq")) else None,
        "ps": round(latest["ps_ttm"], 1) if "ps_ttm" in latest and not pd.isna(latest.get("ps_ttm")) else None,
        "pcf": round(latest["pcf_ttm"], 1) if "pcf_ttm" in latest and not pd.isna(latest.get("pcf_ttm")) else None,
        "roe": round(fin["roe_avg"].iloc[-1] * 100, 1) if fin is not None and "roe_avg" in fin.columns and not pd.isna(fin["roe_avg"].iloc[-1]) else None,
        "avg_amount": round(df["amount"].mean() / 1e8, 1),
        "ret": round(ret_total, 1),
        "sharpe": round(sharpe, 2),
        "avg_turn": round(df["turnover"].mean(), 2) if "turnover" in df.columns else 0,
    }

    # K线
    df_6m = df[df["date"] >= df["date"].max() - pd.Timedelta(days=730)].copy()  # 最近2年

    return jsonify({
        "header": header,
        "metrics": metrics,
        "kline": _build_kline(df_6m, code),
        "valuation": _build_valuation(df),
        "ret_hist": _build_ret_hist(df),
        "roe_chart": _build_roe_chart(fin) if fin is not None else {},
        "growth_chart": _build_growth_chart(fin) if fin is not None else {},
        "fin_table": _build_fin_table(code),
    })


if __name__ == "__main__":
    print("\n  分析报告: http://localhost:8051\n")
    app.run(host="0.0.0.0", port=8051, debug=False)
