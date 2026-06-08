"""
数据拉取进度面板 + 自动看门狗
用法: python progress_dashboard.py
访问: http://localhost:5000

看门狗: 每10秒检查最新文件时间，若超过STALL_TIMEOUT秒未更新则自动重启拉取进程
"""

import sys
import json
import time
import threading
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, jsonify

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"
FIN_DIR = ROOT / "data" / "financial"
FETCH_SCRIPT = ROOT / "scripts" / "fetch_financial_v3.py"
DAILY_UPDATE_SCRIPT = ROOT / "scripts" / "update_daily.py"

app = Flask(__name__)
start_time = datetime.now()
TOTAL_STOCKS = len(list((ROOT / "data" / "daily").glob("*.parquet")))
STALL_TIMEOUT = int(os.environ.get("WATCHDOG_STALL_TIMEOUT", "300"))
GRACE_PERIOD = int(os.environ.get("WATCHDOG_GRACE_PERIOD", "60"))

# 看门狗状态
watchdog_state = {
    "enabled": True,
    "restart_count": 0,
    "last_restart_time": None,
    "last_restart_reason": None,
    "last_file_time": None,
    "last_file_code": None,
    "stall_seconds": 0,
    "fetch_process": None,
    "status": "启动中",
    "grace_until": None,  # 宽限期截止时间
    # 日线更新阶段
    "daily_update_active": False,
    "daily_update_done": False,
    "daily_update_result": None,  # 更新结果摘要
    "daily_latest_date": None,    # 日线数据最新日期
    "daily_target_date": None,    # 目标最新交易日
}

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_latest_fin_file():
    """获取最新财务文件的时间和代码"""
    files = list(FIN_DIR.glob("*.parquet"))
    if not files:
        return None, None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
    return mtime, latest.stem


def start_fetch_process():
    """启动财务拉取子进程 (输出写入日志文件)"""
    python = sys.executable
    log_file = open(LOG_DIR / f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", "w")
    proc = subprocess.Popen(
        [python, str(FETCH_SCRIPT)],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc


def start_daily_update():
    """启动日线增量更新子进程"""
    python = sys.executable
    log_file = open(LOG_DIR / f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", "w")
    proc = subprocess.Popen(
        [python, str(DAILY_UPDATE_SCRIPT)],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc


def check_daily_freshness(full_scan: bool = False):
    """检查日线数据新鲜度, 返回 (latest_date, target_date, fresh_count, stale_count)
    full_scan=False: 抽样检查 (快速, 用于面板展示)
    full_scan=True: 全量扫描 (用于看门狗决策)
    """
    files = sorted(DAILY_DIR.glob("*.parquet"))
    if not files:
        return (None, None, 0, 0)

    target_date = get_daily_target_date()

    if not full_scan and len(files) > 200:
        # 快速抽样: 随机选200只
        import random
        sample = random.sample(files, 200)
    else:
        sample = files

    latest_dates = []
    for f in sample:
        try:
            df = pd.read_parquet(str(f), columns=["date"])
            latest_dates.append(str(df["date"].max())[:10])
        except Exception:
            pass

    if not latest_dates:
        return (None, target_date, 0, 0)

    overall_latest = max(latest_dates)

    if full_scan:
        # 全量统计
        fresh = sum(1 for d in latest_dates if d >= target_date)
        stale = len(latest_dates) - fresh
    else:
        # 抽样推算 (假设样本代表整体)
        fresh_in_sample = sum(1 for d in latest_dates if d >= target_date)
        total = len(files)
        fresh = int(fresh_in_sample / len(latest_dates) * total)
        stale = total - fresh

    return (overall_latest, target_date, fresh, stale)


def get_daily_target_date():
    """获取最近交易日作为日线目标日期
    若当天在15:30之前, 当天数据可能还未发布, 回退到前一交易日
    """
    now = datetime.now()
    today = now
    # 15:30 前, 当日数据可能未发布
    if now.hour < 15 or (now.hour == 15 and now.minute < 30):
        today = now - timedelta(days=1)
    # 周末回退到周五
    wd = today.weekday()
    if wd == 5:
        today = today - timedelta(days=1)
    elif wd == 6:
        today = today - timedelta(days=2)
    return today.strftime("%Y-%m-%d")


def restart_fetch_process(reason: str):
    """重启拉取进程"""
    global watchdog_state
    # 杀掉旧进程
    old = watchdog_state.get("fetch_process")
    if old and old.poll() is None:
        old.kill()
        try:
            old.wait(timeout=5)
        except Exception:
            pass

    # 启动新进程
    print(f"[看门狗] 重启拉取: {reason}")
    proc = start_fetch_process()
    watchdog_state["fetch_process"] = proc
    watchdog_state["restart_count"] += 1
    watchdog_state["last_restart_time"] = datetime.now().strftime("%H:%M:%S")
    watchdog_state["last_restart_reason"] = reason
    watchdog_state["status"] = "运行中"
    watchdog_state["grace_until"] = datetime.now().timestamp() + GRACE_PERIOD
    return proc


def watchdog_loop():
    """看门狗线程: 持续监控文件更新"""
    global watchdog_state

    # 首次启动
    print("[看门狗] 启动拉取进程...")
    proc = start_fetch_process()
    watchdog_state["fetch_process"] = proc
    watchdog_state["grace_until"] = datetime.now().timestamp() + GRACE_PERIOD
    watchdog_state["status"] = "运行中"

    fin_was_done = False  # 追踪财务是否刚完成

    while True:
        time.sleep(10)

        try:
            now_ts = datetime.now().timestamp()
            in_grace = watchdog_state.get("grace_until") and now_ts < watchdog_state["grace_until"]

            last_time, last_code = get_latest_fin_file()
            fin_count = len(list(FIN_DIR.glob("*.parquet")))

            if last_time:
                stall = (datetime.now() - last_time).total_seconds()
                watchdog_state["stall_seconds"] = int(stall)
                watchdog_state["last_file_time"] = last_time.strftime("%H:%M:%S")
                watchdog_state["last_file_code"] = last_code

                # 财务拉取完成 → 检查日线新鲜度, 按需更新
                if fin_count >= TOTAL_STOCKS and not fin_was_done:
                    fin_was_done = True
                    watchdog_state["status"] = "全部完成!"
                    if proc and proc.poll() is None:
                        proc.terminate()

                    # 先检查日线新鲜度 (全量扫描, 确保准确)
                    _, target_date, fresh, stale = check_daily_freshness(full_scan=True)
                    print(f"[看门狗] 财务拉取完成! 日线新鲜度: {fresh}新鲜/{stale}滞后, 目标={target_date}")
                    if stale > 10:  # 超过10只滞后才触发更新
                        print(f"[看门狗] 自动启动日线增量更新...")
                        daily_proc = start_daily_update()
                        watchdog_state["daily_update_active"] = True
                        watchdog_state["daily_update_done"] = False
                        watchdog_state["fetch_process"] = daily_proc
                        watchdog_state["grace_until"] = datetime.now().timestamp() + GRACE_PERIOD
                    else:
                        print(f"[看门狗] 日线数据已足够新鲜, 跳过更新")
                        watchdog_state["daily_update_done"] = True
                        watchdog_state["daily_update_result"] = {
                            "exit_code": 0, "latest_date": target_date,
                            "target_date": target_date, "fresh": fresh, "stale": stale,
                        }
                        watchdog_state["daily_latest_date"] = target_date
                        watchdog_state["daily_target_date"] = target_date

                # 日线更新阶段的监控
                if watchdog_state["daily_update_active"] and not watchdog_state["daily_update_done"]:
                    proc = watchdog_state.get("fetch_process")
                    if proc and proc.poll() is not None:
                        # 日线更新进程已退出
                        exit_code = proc.poll()
                        watchdog_state["daily_update_active"] = False
                        watchdog_state["daily_update_done"] = True
                        freshness = check_daily_freshness()
                        watchdog_state["daily_update_result"] = {
                            "exit_code": exit_code,
                            "latest_date": freshness[0],
                            "target_date": freshness[1],
                            "fresh": freshness[2],
                            "stale": freshness[3],
                        }
                        watchdog_state["daily_latest_date"] = freshness[0]
                        watchdog_state["daily_target_date"] = freshness[1]
                        print(f"[看门狗] 日线更新完成: {freshness[2]}只新鲜, {freshness[3]}只滞后")
                        if freshness[3] > 0:
                            print(f"[看门狗] ⚠️ 仍有{freshness[3]}只日线数据滞后!")
                        break  # 全部任务完成, 退出看门狗循环

                if fin_count >= TOTAL_STOCKS:
                    # 财务已完成, 不再检测僵死
                    continue

                # 僵死检测 (宽限期内跳过)
                if stall > STALL_TIMEOUT and not in_grace and watchdog_state["enabled"]:
                    reason = f"僵死{int(stall)}s (最新: {last_code} @ {last_time.strftime('%H:%M:%S')})"
                    print(f"[看门狗] {reason} -> 自动重启")
                    proc = restart_fetch_process(reason)

            # 检查进程是否还活着
            proc = watchdog_state.get("fetch_process")
            if proc and proc.poll() is not None and not watchdog_state["daily_update_active"]:
                exit_code = proc.poll()
                if exit_code != 0 and fin_count < TOTAL_STOCKS and not in_grace:
                    reason = f"进程退出 (exit={exit_code})"
                    print(f"[看门狗] {reason} -> 自动重启")
                    restart_fetch_process(reason)

            if in_grace:
                remaining = int(watchdog_state["grace_until"] - now_ts)
                watchdog_state["status"] = f"宽限期 {remaining}s"

        except Exception as e:
            print(f"[看门狗] 异常: {e}")


@app.route("/api/progress")
def api_progress():
    daily_count = len(list(DAILY_DIR.glob("*.parquet"))) if DAILY_DIR.exists() else 0
    fin_count = len(list(FIN_DIR.glob("*.parquet"))) if FIN_DIR.exists() else 0

    latest_files = []
    last_time, last_code = get_latest_fin_file()
    if last_time:
        files = sorted(FIN_DIR.glob("*.parquet"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[:10]:
            latest_files.append({
                "code": f.stem,
                "time": datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S"),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })

    elapsed = datetime.now() - start_time
    hours, rem = divmod(int(elapsed.total_seconds()), 3600)
    minutes = rem // 60

    eta_str = "--"
    rate_str = "--"
    if fin_count > 0 and fin_count < TOTAL_STOCKS:
        elapsed_min = max(elapsed.total_seconds() / 60, 0.1)
        rate = fin_count / elapsed_min
        remaining = TOTAL_STOCKS - fin_count
        eta_min = remaining / max(rate, 0.01)
        eta_h = int(eta_min // 60)
        eta_m = int(eta_min % 60)
        eta_str = f"{eta_h}h{eta_m}m"
        rate_str = f"{rate:.1f} 只/分钟"

    # 日线新鲜度
    daily_fresh_info = {}
    if watchdog_state["daily_update_done"] and watchdog_state["daily_update_result"]:
        r = watchdog_state["daily_update_result"]
        daily_fresh_info = {
            "fresh": r["fresh"], "stale": r["stale"],
            "latest_date": r["latest_date"], "target_date": r["target_date"],
            "exit_code": r["exit_code"],
        }
    elif watchdog_state["daily_update_active"]:
        daily_fresh_info = {"status": "updating", "latest_date": watchdog_state.get("daily_latest_date"),
                           "target_date": watchdog_state.get("daily_target_date")}
    else:
        try:
            latest_date, target_date, fresh, stale = check_daily_freshness()
            daily_fresh_info = {
                "fresh": fresh, "stale": stale,
                "latest_date": latest_date, "target_date": target_date,
            }
            watchdog_state["daily_latest_date"] = latest_date
            watchdog_state["daily_target_date"] = target_date
        except Exception:
            daily_fresh_info = {"fresh": 0, "stale": 0, "latest_date": None, "target_date": None}

    return jsonify({
        "daily": {"count": daily_count, "total": TOTAL_STOCKS,
                  "pct": round(daily_count / TOTAL_STOCKS * 100, 1),
                  "done": daily_count >= TOTAL_STOCKS},
        "financial": {"count": fin_count, "total": TOTAL_STOCKS,
                      "pct": round(fin_count / TOTAL_STOCKS * 100, 1),
                      "done": fin_count >= TOTAL_STOCKS},
        "latest_files": latest_files,
        "elapsed": f"{int(hours)}h{int(minutes)}m",
        "eta": eta_str,
        "rate": rate_str,
        "time": datetime.now().strftime("%H:%M:%S"),
        "watchdog": {
            "status": watchdog_state["status"],
            "restart_count": watchdog_state["restart_count"],
            "last_restart": watchdog_state["last_restart_time"] or "--",
            "last_restart_reason": watchdog_state["last_restart_reason"] or "--",
            "last_file_time": watchdog_state["last_file_time"] or "--",
            "last_file_code": watchdog_state["last_file_code"] or "--",
            "stall_seconds": watchdog_state["stall_seconds"],
            "stall_warning": watchdog_state["stall_seconds"] > STALL_TIMEOUT and fin_count < TOTAL_STOCKS,
        },
        "daily_freshness": daily_fresh_info,
        "daily_update_active": watchdog_state["daily_update_active"] and not watchdog_state["daily_update_done"],
    })


@app.route("/api/restart", methods=["POST"])
def api_restart():
    """手动重启（需要 token 验证）"""
    token = request.headers.get("X-Auth-Token", "")
    expected = os.environ.get("PROGRESS_DASHBOARD_TOKEN")
    if not expected:
        return jsonify({"ok": False, "error": "未配置 PROGRESS_DASHBOARD_TOKEN 环境变量"}), 500
    if token != expected:
        return jsonify({"ok": False, "error": "未授权"}), 403
    restart_fetch_process("手动重启")
    return jsonify({"ok": True, "time": datetime.now().strftime("%H:%M:%S")})


@app.route("/")
def index():
    return HTML


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-cache"
    return response


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>数据拉取进度</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Microsoft YaHei', sans-serif; background: #0d1117; color: #c9d1d9; padding: 24px; }
h1 { text-align: center; color: #58a6ff; margin-bottom: 6px; font-size: 1.4em; }
.subtitle { text-align: center; color: #8b949e; margin-bottom: 24px; font-size: 0.85em; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; max-width: 1000px; margin: 0 auto; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }
.card h2 { font-size: 1em; margin-bottom: 14px; color: #f0f6fc; display: flex; align-items: center; gap: 8px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot.ok { background: #3fb950; box-shadow: 0 0 8px #3fb950; animation: pulse 1.5s infinite; }
.dot.done { background: #58a6ff; animation: none; }
.dot.stall { background: #d29922; box-shadow: 0 0 8px #d29922; animation: pulse 0.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.big { font-size: 2.6em; font-weight: bold; color: #f0f6fc; }
.small { font-size: 0.35em; color: #8b949e; }
.bar-wrap { background: #21262d; border-radius: 6px; height: 16px; margin: 10px 0; overflow: hidden; }
.bar { height: 100%; border-radius: 6px; transition: width 0.8s; }
.bar.blue { background: linear-gradient(90deg, #1f6feb, #58a6ff); }
.bar.green { background: linear-gradient(90deg, #238636, #3fb950); }
.pct { text-align: right; color: #8b949e; font-size: 0.85em; }
.info { color: #8b949e; font-size: 0.85em; margin-top: 12px; line-height: 1.7; }
.info b { color: #f0f6fc; }
.info .warn { color: #d29922; font-weight: bold; }
.watchdog { border: 1px solid #30363d; }
.stall-badge { display: inline-block; background: #d29922; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-left: 8px; }
button { background: #238636; color: #fff; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85em; margin-top: 8px; }
button:hover { background: #2ea043; }
button:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; }
.latest { max-width: 1000px; margin: 20px auto 0; }
.latest h2 { color: #f0f6fc; margin-bottom: 10px; font-size: 1em; }
.latest table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
.latest th, .latest td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #21262d; }
.latest th { color: #8b949e; font-weight: normal; }
.latest code { color: #58a6ff; background: #1c2128; padding: 1px 6px; border-radius: 3px; }
.footer { text-align: center; color: #484f58; margin-top: 20px; font-size: 0.78em; }
</style>
</head>
<body>
<h1>A股数据拉取进度</h1>
<p class="subtitle">日线行情 & 财务数据 · 自动监控 · <span id="refresh-hint">3秒刷新</span></p>

<div class="grid">
  <div class="card">
    <h2><span id="daily-dot" class="dot"></span> 日线行情</h2>
    <div class="big"><span id="daily-count">--</span><span class="small"> / <span id="daily-total">--</span></span></div>
    <div class="bar-wrap"><div id="daily-bar" class="bar blue" style="width:0%"></div></div>
    <div class="pct" id="daily-pct">--</div>
    <div class="info" style="margin-top:10px">
      数据最新日期: <b id="daily-latest-date">--</b><br>
      目标日期: <b id="daily-target-date">--</b><br>
      已更新: <b id="daily-fresh">--</b> / 滞后: <b id="daily-stale">--</b>
      <span id="daily-update-badge"></span>
    </div>
  </div>
  <div class="card">
    <h2><span id="fin-dot" class="dot"></span> 财务数据</h2>
    <div class="big"><span id="fin-count">--</span><span class="small"> / <span id="fin-total">--</span></span></div>
    <div class="bar-wrap"><div id="fin-bar" class="bar green" style="width:0%"></div></div>
    <div class="pct" id="fin-pct">--</div>
    <div class="info">
      速率: <b id="rate">--</b><br>
      已运行: <b id="elapsed">--</b><br>
      预计剩余: <b id="eta">--</b><br>
      更新: <b id="time">--</b>
    </div>
  </div>
  <div class="card watchdog">
    <h2>看门狗 <span id="wd-status-badge"></span></h2>
    <div class="info">
      状态: <b id="wd-status">--</b><br>
      最新文件: <code id="wd-file">--</code> @ <b id="wd-filetime">--</b><br>
      停滞时长: <b id="wd-stall">--</b><br>
      自动重启: <b id="wd-restarts">--</b> 次<br>
      上次重启原因: <span id="wd-reason">--</span>
    </div>
    <button id="btn-restart" onclick="manualRestart()">手动重启拉取</button>
  </div>
</div>

<div class="latest">
  <h2>最新拉取 (财务)</h2>
  <table>
    <thead><tr><th>代码</th><th>时间</th><th>大小</th></tr></thead>
    <tbody id="latest-tbody"></tbody>
  </table>
</div>

<p class="footer">quant-learning · 看门狗自动监控</p>

<script>
function fmt(n) { return n.toLocaleString(); }
let count = 0;

async function refresh() {
  try {
    let r = await fetch('/api/progress');
    let d = await r.json();

    // 日线
    document.getElementById('daily-count').textContent = fmt(d.daily.count);
    document.getElementById('daily-total').textContent = fmt(d.daily.total);
    document.getElementById('daily-bar').style.width = d.daily.pct + '%';
    document.getElementById('daily-pct').textContent = d.daily.pct + '% ' + (d.daily.done ? '✅' : '');
    document.getElementById('daily-dot').className = 'dot ' + (d.daily.done ? 'done' : 'ok');

    // 日线新鲜度
    if (d.daily_freshness) {
      document.getElementById('daily-latest-date').textContent = d.daily_freshness.latest_date || '--';
      document.getElementById('daily-target-date').textContent = d.daily_freshness.target_date || '--';
      if (d.daily_update_active) {
        document.getElementById('daily-fresh').textContent = '更新中...';
        document.getElementById('daily-stale').textContent = '更新中...';
        document.getElementById('daily-update-badge').innerHTML = '<span class="stall-badge" style="background:#1f6feb">日线更新中</span>';
      } else {
        document.getElementById('daily-fresh').textContent = fmt(d.daily_freshness.fresh || 0);
        document.getElementById('daily-stale').textContent = fmt(d.daily_freshness.stale || 0);
        let allFresh = d.daily_freshness.stale === 0 && d.daily_freshness.fresh > 0;
        document.getElementById('daily-update-badge').innerHTML = allFresh
          ? '<span class="stall-badge" style="background:#3fb950">全部最新</span>'
          : (d.daily_freshness.stale > 0
            ? '<span class="stall-badge">' + d.daily_freshness.stale + '只滞后</span>'
            : '');
      }
    }

    // 财务
    let finDone = d.financial.done;
    let stall = d.watchdog.stall_warning && !finDone;
    document.getElementById('fin-count').textContent = fmt(d.financial.count);
    document.getElementById('fin-total').textContent = fmt(d.financial.total);
    document.getElementById('fin-bar').style.width = d.financial.pct + '%';
    document.getElementById('fin-pct').textContent = d.financial.pct + '% ' + (finDone ? '✅' : stall ? '⚠️ 僵死' : '');
    document.getElementById('fin-dot').className = 'dot ' + (finDone ? 'done' : stall ? 'stall' : 'ok');

    document.getElementById('rate').textContent = d.rate;
    document.getElementById('elapsed').textContent = d.elapsed;
    document.getElementById('eta').textContent = d.eta;
    document.getElementById('time').textContent = d.time;

    // 看门狗
    let w = d.watchdog;
    let badge = document.getElementById('wd-status-badge');
    if (w.status.includes('完成')) badge.innerHTML = '<span class="stall-badge" style="background:#3fb950">完成</span>';
    else if (w.stall_warning) badge.innerHTML = '<span class="stall-badge">僵死!</span>';
    else badge.innerHTML = '<span style="color:#3fb950;font-size:0.8em;">● 活跃</span>';

    document.getElementById('wd-status').textContent = w.status;
    document.getElementById('wd-file').textContent = w.last_file_code;
    document.getElementById('wd-filetime').textContent = w.last_file_time;
    let stallMin = Math.floor(w.stall_seconds / 60);
    let stallSec = w.stall_seconds % 60;
    let stallStr = stallMin > 0 ? stallMin + 'm' + stallSec + 's' : stallSec + 's';
    if (w.stall_warning) stallStr = '⚠️ ' + stallStr;
    document.getElementById('wd-stall').innerHTML = stallStr;
    document.getElementById('wd-restarts').textContent = w.restart_count;
    document.getElementById('wd-reason').textContent = w.last_restart_reason;

    // 最新文件
    let tb = document.getElementById('latest-tbody');
    tb.innerHTML = d.latest_files.map(f =>
      `<tr><td><code>${f.code}</code></td><td>${f.time}</td><td>${f.size_kb} KB</td></tr>`
    ).join('');

    count++;
    document.getElementById('refresh-hint').textContent = '刷新 #' + count;
  } catch(e) {
    document.getElementById('refresh-hint').textContent = '连接失败';
  }
}

async function manualRestart() {
  let btn = document.getElementById('btn-restart');
  btn.disabled = true;
  btn.textContent = '重启中...';
  try {
    await fetch('/api/restart', {method:'POST'});
    setTimeout(() => { btn.disabled = false; btn.textContent = '手动重启拉取'; }, 5000);
  } catch(e) {
    btn.disabled = false;
    btn.textContent = '手动重启拉取';
  }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    # 启动看门狗线程
    watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
    watchdog_thread.start()

    print(f"进度面板: http://localhost:5000")
    print(f"看门狗: 每10秒检查 | 僵死阈值: {STALL_TIMEOUT}s | 自动重启已启用")
    app.run(host="0.0.0.0", port=5000, debug=False)
