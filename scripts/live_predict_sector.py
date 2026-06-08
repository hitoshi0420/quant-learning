"""
行业维度策略预测 — 策略A(行业集中) + 策略B(跨行业龙头)
用法:
  python scripts/live_predict_sector.py          # 两套策略都运行
  python scripts/live_predict_sector.py --mode a   # 只运行策略A
  python scripts/live_predict_sector.py --mode b   # 只运行策略B
"""

import sys
import io
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from live_engine import LiveEngine


def run_strategy_a(top_n: int = 8):
    """策略A: 行业集中"""
    engine = LiveEngine()

    print("=" * 70)
    print("  策略A: 行业集中")
    print("  逻辑: 先选最强行业 → 再在该行业内精选个股")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print("\n[1/4] 加载数据 + 计算因子...")
    engine.load_data()
    engine.compute_factors()

    print("\n[2/4] IC分析 + 多策略打分...")
    engine.estimate_ic()
    engine.score()

    print(f"\n[3/4] 行业筛选 + 集中选股 (top_{top_n})...")
    engine.build_sector_concentration(top_n=top_n)

    print("\n[4/4] 输出...")
    engine.print_picks()
    engine.save("live_picks_sector.csv")

    return engine


def run_strategy_b(top_n: int = 8):
    """策略B: 跨行业龙头"""
    engine = LiveEngine()

    print("=" * 70)
    print("  策略B: 跨行业龙头")
    print("  逻辑: 每个行业选1支冠军 → 跨行业分散组合")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print("\n[1/4] 加载数据 + 计算因子...")
    engine.load_data()
    engine.compute_factors()

    print("\n[2/4] IC分析 + 多策略打分...")
    engine.estimate_ic()
    engine.score()

    print(f"\n[3/4] 各行业选龙头 + 跨行业组合 (top_{top_n})...")
    engine.build_cross_industry(top_n=top_n)

    print("\n[4/4] 输出...")
    engine.print_picks()
    engine.save("live_picks_cross.csv")

    return engine


def main():
    mode = "both"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1].lower()

    if mode in ("a", "both"):
        engine_a = run_strategy_a(top_n=8)
        if mode == "both":
            print("\n\n")

    if mode in ("b", "both"):
        engine_b = run_strategy_b(top_n=8)


if __name__ == "__main__":
    main()
