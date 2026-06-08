"""
全局配置 — 所有可调参数集中管理
"""

from pathlib import Path

# === 路径 ===
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
FINANCIAL_DIR = DATA_DIR / "financial"
LOG_DIR = ROOT / "logs"

DAILY_DIR.mkdir(parents=True, exist_ok=True)
FINANCIAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === 数据拉取参数 ===
START_DATE = "20200101"     # 注意：YYYYMMDD 格式，5年数据够学习和回测
REQUEST_DELAY = 0.2         # 两次请求间隔（秒）
MAX_WORKERS = 3             # 并发线程数（Baostock 建议 ≤ 3）
RETRY_TIMES = 2             # 单只股票重试次数

# === 过滤规则 ===
MIN_LISTED_DAYS = 252       # 上市至少满 1 年（约 252 个交易日）
EXCLUDE_ST = True           # 排除 ST
EXCLUDE_NEW_SHARES = True   # 排除次新股（上市不满 1 年）

# === 多因子策略参数 ===

# 流动性筛选
LIQUIDITY_TOP_N = 800           # 选成交额最大的 N 只股票
LIQUIDITY_LOOKBACK_DAYS = 90    # 因子计算的历史数据天数
MIN_DAILY_AMOUNT = 30_000_000   # 最低日均成交额（3000万，全市场）

# 因子权重约束（分组限详见 factor_library.MAX_GROUP_WEIGHT）
MAX_SINGLE_FACTOR_WEIGHT = 0.20 # 单因子最大权重
FACTOR_CORRELATION_THRESHOLD = 0.7  # 去相关阈值

# IC估计
IC_LOOKBACK_MONTHS = 36         # 滚动窗口月数
IC_DECAY_HALFLIFE = 12          # 指数衰减半衰期（月）
MIN_IC_IR_THRESHOLD = 0.08      # 显著因子阈值

# 回测
REFIT_EVERY_MONTHS = 12         # IC 重估计间隔（月）
TOP_N_STOCKS = 30               # 持仓股票数（strategy_runner 使用）

# === 风控参数 ===
RISK_PARAMS = {
    "max_single_weight": 0.10,      # 单只股票最大权重
    "min_single_weight": 0.01,      # 单只股票最小权重
    "max_industry_weight": 0.25,    # 单一申万行业最大权重
    "stop_loss": -0.15,             # 个股止损线
    "portfolio_stop": -0.10,        # 组合回撤止损线
    "vol_target": 0.15,             # 目标年化波动率
    "min_cash_buffer": 0.05,        # 最低现金保留比例
    "max_turnover": 0.50,           # 月换手率上限
}

# === 交易成本 ===
COMMISSION_RATE = 0.0003    # 万三佣金
STAMP_DUTY = 0.0005         # 千0.5印花税（仅卖出）
SLIPPAGE = 0.001            # 0.1% 滑点

# === 财务数据字段映射（Baostock 查询定义） ===
FINANCIAL_FIELDS = {
    "profit": {
        "fields": "roeAvg,npMargin,gpMargin,netProfit,epsTTM,MBRevenue,totalShare,liqaShare",
        "columns": ["roe_avg", "np_margin", "gp_margin", "net_profit", "eps_ttm",
                    "mb_revenue", "total_share", "liqa_share"],
    },
    "balance": {
        "fields": "currentRatio,quickRatio,cashRatio,YOYLiability,liabilityToAsset,assetToEquity",
        "columns": ["current_ratio", "quick_ratio", "cash_ratio", "yoy_liability",
                    "liability_to_asset", "asset_to_equity"],
    },
    "cash_flow": {
        "fields": "CAToAsset,NCAToAsset,tangibleAssetToAsset,ebitToInterest,CFOToOR,CFOToNP,CFOToGr",
        "columns": ["ca_to_asset", "nca_to_asset", "tangible_asset_ratio", "ebit_to_interest",
                    "cfo_to_or", "cfo_to_np", "cfo_to_gr"],
    },
    "growth": {
        "fields": "YOYEquity,YOYAsset,YOYNI,YOYEPSBasic,YOYPNI",
        "columns": ["yoy_equity", "yoy_asset", "yoy_ni", "yoy_eps", "yoy_pni"],
    },
    "operation": {
        "fields": "NRTurnRatio,NRTurnDays,INVTurnRatio,INVTurnDays,CATurnRatio,AssetTurnRatio",
        "columns": ["nr_turn_ratio", "nr_turn_days", "inv_turn_ratio", "inv_turn_days",
                    "ca_turn_ratio", "asset_turn_ratio"],
    },
    "dupont": {
        "fields": "dupontROE,dupontAssetStoEquity,dupontAssetTurn,dupontPnitoni,"
                  "dupontNitogr,dupontTaxBurden,dupontIntburden,dupontEbittogr",
        "columns": ["dupont_roe", "dupont_asset_to_equity", "dupont_asset_turn",
                    "dupont_pni_to_ni", "dupont_ni_to_gr", "dupont_tax_burden",
                    "dupont_int_burden", "dupont_ebit_to_gr"],
    },
}

# === 缓存配置 ===
CACHE_DIR = ROOT / "data" / "cache"
CACHE_CONFIG = {
    "daily_factors":        {"max_age_days": 1},
    "ic_estimates":         {"max_age_days": 7},
    "industry_map":         {"max_age_days": 30},
    "stock_names":          {"max_age_days": 7},
    "monthly_factor_table": {"max_age_days": 30},
}
