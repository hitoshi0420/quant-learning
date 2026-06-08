"""
数据浏览器 — 在命令行里快速查看已拉取的股票数据

用法:
    python explore.py                # 列出摘要
    python explore.py 000001         # 查看平安银行
    python explore.py 600519         # 查看贵州茅台
"""

import sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "daily"


def list_summary():
    """列出所有股票摘要"""
    files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"\n{'代码':<8} {'行数':>6}  {'起始':>12}  {'截止':>12}  {'最新收盘':>10}")
    print("-" * 58)

    for f in files:
        df = pd.read_parquet(f, columns=["date", "close"])
        code = f.stem
        print(f"{code:<8} {len(df):>6}  {str(df['date'].min().date()):>12}  "
              f"{str(df['date'].max().date()):>12}  {df['close'].iloc[-1]:>10.2f}")

    print(f"\n共 {len(files)} 只股票")


def view_one(code: str):
    """查看单只股票详情"""
    fpath = DATA_DIR / f"{code}.parquet"
    if not fpath.exists():
        print(f"❌ 找不到 {code}")
        return

    df = pd.read_parquet(fpath)
    print(f"\n{'='*60}")
    print(f"  {code}  日期: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"{'='*60}")

    # 最近 20 天
    print("\n📊 最近 20 个交易日:")
    recent = df.tail(20)[["date", "open", "high", "low", "close", "volume", "pct_change"]]
    recent["date"] = recent["date"].dt.date
    print(recent.to_string(index=False))

    # 简要统计
    print(f"\n📈 简要统计:")
    print(f"  区间收益: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.1f}%")
    print(f"  最高价: {df['high'].max():.2f}")
    print(f"  最低价: {df['low'].min():.2f}")
    print(f"  日均成交额: {df['amount'].mean() / 1e8:.1f} 亿")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_one(sys.argv[1])
    else:
        list_summary()
