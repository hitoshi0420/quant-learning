"""
实时因子选股 — 基于最新日线数据 (无未来信息)
用法: python scripts/live_predict.py

v4.0 — 多策略并行集成: 价值防御 + 成长进攻 + 质量均衡
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


def main():
    print("=" * 70)
    print(f"  量化多因子策略 v4.0 — 多策略并行预测")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  策略簇: 价值防御 + 成长进攻 + 质量均衡")
    print("=" * 70)

    engine = LiveEngine()

    print("\n[1/5] 加载数据...")
    engine.load_data(lookback_days=90)

    print("\n[2/5] 计算因子...")
    engine.compute_factors()

    print("\n[3/5] 多策略IC分析 + 因子选择...")
    engine.estimate_ic()

    print("\n[4/5] 多策略打分...")
    engine.score(neutralize=False)

    print("\n[5/5] 多簇集成 + 组合构建 + 风控...")
    engine.build()

    engine.print_picks()
    engine.save()

    print(f"\n{'=' * 70}")
    print(f"  v4.0 预测完成")


if __name__ == "__main__":
    main()
