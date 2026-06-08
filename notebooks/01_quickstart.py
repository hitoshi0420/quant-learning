"""
============================================================
快速上手：5 分钟从拉数据到第一张收益曲线
============================================================

运行方式: python notebooks/01_quickstart.py
（也可以复制代码到 Jupyter 逐格运行）

前提: 先安装依赖
    pip install -r requirements.txt
"""

# %% [markdown]
# ## Step 0: 导入 & 切换到项目根目录

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## Step 1: 拉数据（第一次运行 ~15 分钟，之后增量几秒）

print("Step 1: 拉取数据...")


# 正式使用时导入 pipeline 模块
# from data_fetcher import get_stock_list, fetch_all_daily

# 如果只想快速测试，用 akshare 直接拉一只股票：
# import akshare as ak
# df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20200101", end_date="20250524", adjust="qfq")
# print(df.head())

print("提示: 取消注释上面的 import 语句来真正运行")

# %% [markdown]
# ## Step 2: 加载本地数据

print("Step 2: 加载本地数据...")

# from data_fetcher import load_all_daily
# full = load_all_daily()
# print(f"数据形状: {full.shape}")

# %% [markdown]
# ## Step 3: 计算收益率

# from data_fetcher import add_returns, add_ma, add_atr
# full = add_returns(full)
# full = add_ma(full)
# full = add_atr(full)

# %% [markdown]
# ## Step 4: 画第一张图 — 各年份股票数量 vs 日均收益率

def plot_universe_overview(df: pd.DataFrame):
    """全市场概览：各年份股票数量和日均收益率"""
    df = df.copy()
    df["year"] = df["date"].dt.year

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左: 每年股票数量
    yearly_counts = df.groupby("year")["code"].nunique()
    axes[0].bar(yearly_counts.index, yearly_counts.values, color="steelblue")
    axes[0].set_title("各年份股票数量")
    axes[0].set_xlabel("年份")

    # 中: 每年日均收益率 (bp)
    yearly_ret = df.groupby("year").apply(
        lambda g: g["ret"].mean() * 10000 if "ret" in g.columns else np.nan
    )
    axes[1].bar(yearly_ret.index, yearly_ret.values, color="darkorange")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("各年份日均收益率 (bp)")
    axes[1].set_xlabel("年份")

    # 右: 收益率分布直方图
    if "ret" in df.columns:
        ret_sample = df["ret"].dropna().sample(min(100_000, len(df)))
        axes[2].hist(ret_sample * 100, bins=100, color="gray", alpha=0.7)
        axes[2].axvline(0, color="red", linewidth=0.5)
        axes[2].set_title("日收益率分布")
        axes[2].set_xlabel("收益率 (%)")

    plt.tight_layout()
    plt.savefig("output/universe_overview.png", dpi=150, bbox_inches="tight")
    plt.show()


# plot_universe_overview(full)

# %% [markdown]
# ## Step 5: 写一个最简单的均线策略回测

def simple_sma_strategy(df: pd.DataFrame, short=5, long=20, capital=100_000):
    """
    最简单的双均线策略：
    - 快线上穿慢线 → 全仓买入
    - 快线下穿慢线 → 全部卖出
    """
    df = df.sort_values("date").copy()

    # 信号
    df["signal"] = 0
    df.loc[df[f"ma_{short}"] > df[f"ma_{long}"], "signal"] = 1
    df["position"] = df["signal"].shift(1)  # 次日开盘执行

    # 策略收益
    df["strat_ret"] = df["position"] * df["ret"]

    # 累计净值
    df["benchmark"] = (1 + df["ret"].fillna(0)).cumprod()
    df["strategy"] = (1 + df["strat_ret"].fillna(0)).cumprod()

    return df


# 测试单只股票
# pab_df = full[full["code"] == "000001"].copy()
# result = simple_sma_strategy(pab_df)
#
# plt.figure(figsize=(14, 5))
# plt.plot(result["date"], result["benchmark"], label="买入持有", alpha=0.7)
# plt.plot(result["date"], result["strategy"], label="双均线策略", linewidth=1.5)
# plt.legend()
# plt.title("000001 平安银行 — 双均线策略回测")
# plt.show()

# %% [markdown]
# ## 总结

print("""
文件结构:
  scripts/
    ├── config.py        # 全局参数
    ├── data_fetcher.py  # 数据拉取 & 本地读写
    └── pipeline.py      # 每日自动更新入口
  data/
    └── daily/           # 每只股票一个 parquet 文件
  notebooks/
    └── 01_quickstart.py # 本文件

下一步:
  1. pip install -r requirements.txt
  2. python scripts/pipeline.py  # 拉全市场数据
  3. 打开 Jupyter，开始写策略
""")
