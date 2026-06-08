"""
统一日志配置 — 替代散落的 print() 调用

用法:
    from logger import logger
    logger.info("加载数据...")
    logger.warning("数据不足，使用默认值")
    logger.error("计算失败: {}", e)
"""
import sys
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 移除默认 handler
logger.remove()

# 控制台输出 — 简洁格式
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# 文件输出 — 完整日志
logger.add(
    LOG_DIR / "quant_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    encoding="utf-8",
    enqueue=True,  # 线程安全异步写入
)
