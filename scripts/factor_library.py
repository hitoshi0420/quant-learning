"""
因子注册中心 — 单一定义所有因子、分组、方向、数据源
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FactorSource(Enum):
    DAILY = "daily"          # 从日线数据计算
    FINANCIAL = "financial"   # 从财报数据映射
    DERIVED = "derived"       # 从其他因子派生


class FactorDirection(Enum):
    POSITIVE = 1    # 因子值越大越好
    NEGATIVE = -1   # 因子值越小越好


@dataclass
class FactorDefinition:
    name: str                          # factor_xxx
    short_name: str                    # xxx (去掉 factor_ 前缀)
    display_name: str                  # 中文名
    group: str                         # 分组 (value/momentum/volatility/liquidity/profitability/growth/quality/reversal)
    source: FactorSource               # 数据来源
    direction: FactorDirection         # 因子方向
    daily_col: Optional[str] = None    # 日线数据列名（source=DAILY 时）
    fin_col: Optional[str] = None      # 财报数据列名（source=FINANCIAL 时）
    calculation: Optional[str] = None  # 计算公式描述
    needs_sign_flip: bool = False      # 是否需要取负（如波动率、负债率）


# ============================================================
# 因子定义库（22个因子 + size）
# ============================================================

FACTOR_DEFINITIONS: dict[str, FactorDefinition] = {
    # ---- 估值因子 (Value) ----
    "factor_ep": FactorDefinition(
        name="factor_ep", short_name="ep", display_name="盈利收益率(EP)",
        group="value", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        daily_col="pe_ttm", calculation="1 / pe_ttm (剔除负值)"),
    "factor_bp": FactorDefinition(
        name="factor_bp", short_name="bp", display_name="账面市值比(BP)",
        group="value", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        daily_col="pb_mrq", calculation="1 / pb_mrq"),
    "factor_sp": FactorDefinition(
        name="factor_sp", short_name="sp", display_name="收入市值比(SP)",
        group="value", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        daily_col="ps_ttm", calculation="1 / ps_ttm"),
    "factor_cfp": FactorDefinition(
        name="factor_cfp", short_name="cfp", display_name="现金市值比(CFP)",
        group="value", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        daily_col="pcf_ttm", calculation="1 / pcf_ttm"),

    # ---- 动量因子 (Momentum) ----
    "factor_momentum_1m": FactorDefinition(
        name="factor_momentum_1m", short_name="momentum_1m", display_name="短期动量(1月)",
        group="momentum", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        calculation="ret_1m (近1月收益)"),
    "factor_momentum_3m": FactorDefinition(
        name="factor_momentum_3m", short_name="momentum_3m", display_name="中期动量(3月)",
        group="momentum", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        calculation="ret_3m (近3月收益)"),
    "factor_momentum_12m1m": FactorDefinition(
        name="factor_momentum_12m1m", short_name="momentum_12m1m", display_name="经典动量(12-1月)",
        group="momentum", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        calculation="(1+ret_12m)/(1+ret_1m)-1, 跳过最近1个月"),
    "factor_pe_momentum": FactorDefinition(
        name="factor_pe_momentum", short_name="pe_momentum", display_name="PE动量",
        group="momentum", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        daily_col="pe_ttm", calculation="-pe_change_1m (PE收缩→好)", needs_sign_flip=True),

    # ---- 波动率因子 (Volatility) ----
    "factor_vol_20d": FactorDefinition(
        name="factor_vol_20d", short_name="vol_20d", display_name="波动率(20日)",
        group="volatility", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        calculation="-std(ret_1d, 20)", needs_sign_flip=True),
    "factor_vol_60d": FactorDefinition(
        name="factor_vol_60d", short_name="vol_60d", display_name="波动率(60日)",
        group="volatility", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        calculation="-std(ret_1d, 60)", needs_sign_flip=True),

    # ---- 流动性因子 (Liquidity) ----
    "factor_turnover": FactorDefinition(
        name="factor_turnover", short_name="turnover", display_name="换手率",
        group="liquidity", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        daily_col="turnover", calculation="-turnover_20d (低换手→高预期收益)", needs_sign_flip=True),
    "factor_abnormal_turn": FactorDefinition(
        name="factor_abnormal_turn", short_name="abnormal_turn", display_name="异常换手",
        group="liquidity", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        calculation="-(turnover_5d/turnover_20d - 1)", needs_sign_flip=True),

    # ---- 反转因子 (Reversal) ----
    "factor_reversal": FactorDefinition(
        name="factor_reversal", short_name="reversal", display_name="短期反转",
        group="reversal", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        calculation="-ret_5d (跌得多→预期反弹)", needs_sign_flip=True),

    # ---- 规模因子 ----
    "factor_size": FactorDefinition(
        name="factor_size", short_name="size", display_name="规模(Size)",
        group="liquidity", source=FactorSource.DAILY, direction=FactorDirection.POSITIVE,
        daily_col="amount", calculation="-log(amount_20d) (小市值效应)", needs_sign_flip=True),

    # ---- 盈利因子 (Profitability) ----
    "factor_roe": FactorDefinition(
        name="factor_roe", short_name="roe", display_name="ROE",
        group="profitability", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="roe_avg"),
    "factor_np_margin": FactorDefinition(
        name="factor_np_margin", short_name="np_margin", display_name="净利润率",
        group="profitability", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="np_margin"),
    "factor_gp_margin": FactorDefinition(
        name="factor_gp_margin", short_name="gp_margin", display_name="毛利率",
        group="profitability", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="gp_margin"),

    # ---- 成长因子 (Growth) ----
    "factor_yoy_ni": FactorDefinition(
        name="factor_yoy_ni", short_name="yoy_ni", display_name="净利润增速",
        group="growth", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="yoy_ni"),
    "factor_yoy_equity": FactorDefinition(
        name="factor_yoy_equity", short_name="yoy_equity", display_name="净资产增速",
        group="growth", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="yoy_equity"),

    # ---- 质量因子 (Quality) ----
    "factor_current_ratio": FactorDefinition(
        name="factor_current_ratio", short_name="current_ratio", display_name="流动比率",
        group="quality", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="current_ratio"),
    "factor_quick_ratio": FactorDefinition(
        name="factor_quick_ratio", short_name="quick_ratio", display_name="速动比率",
        group="quality", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="quick_ratio"),
    "factor_liability_to_asset": FactorDefinition(
        name="factor_liability_to_asset", short_name="liability_to_asset", display_name="资产负债率",
        group="quality", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="liability_to_asset", calculation="取负(高负债→高风险)", needs_sign_flip=True),
    "factor_asset_turn": FactorDefinition(
        name="factor_asset_turn", short_name="asset_turn", display_name="资产周转率",
        group="quality", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="asset_turn_ratio"),
    "factor_cfo_to_np": FactorDefinition(
        name="factor_cfo_to_np", short_name="cfo_to_np", display_name="现金流/净利润",
        group="quality", source=FactorSource.FINANCIAL, direction=FactorDirection.POSITIVE,
        fin_col="cfo_to_np"),
}


# ============================================================
# 因子分组定义
# ============================================================

FACTOR_GROUPS: dict[str, list[str]] = {
    "value":        ["factor_ep", "factor_bp", "factor_sp", "factor_cfp"],
    "momentum":     ["factor_momentum_1m", "factor_momentum_3m",
                     "factor_momentum_12m1m", "factor_pe_momentum"],
    "volatility":   ["factor_vol_20d", "factor_vol_60d"],
    "liquidity":    ["factor_turnover", "factor_abnormal_turn", "factor_size"],
    "profitability": ["factor_roe", "factor_np_margin", "factor_gp_margin"],
    "growth":       ["factor_yoy_ni", "factor_yoy_equity"],
    "quality":      ["factor_current_ratio", "factor_quick_ratio",
                     "factor_liability_to_asset", "factor_asset_turn", "factor_cfo_to_np"],
    "reversal":     ["factor_reversal"],
}

GROUP_DISPLAY_NAMES: dict[str, str] = {
    "value": "估值", "momentum": "动量", "volatility": "波动率",
    "liquidity": "流动性", "profitability": "盈利", "growth": "成长",
    "quality": "质量", "reversal": "反转",
}

# 组权重上限（防止单一风格主导）
MAX_GROUP_WEIGHT: dict[str, float] = {
    "value": 0.25, "momentum": 0.25, "volatility": 0.15,
    "liquidity": 0.15, "profitability": 0.15, "growth": 0.20,
    "quality": 0.20, "reversal": 0.10,
}

# 单因子权重上限
MAX_SINGLE_FACTOR_WEIGHT = 0.20

# 因子相关度阈值（超过则去重）
CORRELATION_THRESHOLD = 0.7

# 策略簇定义 — 三簇并行，覆盖不同风格
STRATEGY_CLUSTERS = {
    "价值防御": {
        "factor_groups": ["value", "quality"],
        "pick_count": 3,
        "description": "低估值+高质量，挖掘被低估的优质资产",
        "style": "防御型",
    },
    "成长进攻": {
        "factor_groups": ["momentum", "growth"],
        "pick_count": 3,
        "description": "动量+成长，捕捉趋势和成长机会(含AI/芯片/科技)",
        "style": "进攻型",
    },
    "质量均衡": {
        "factor_groups": ["profitability", "liquidity", "volatility", "reversal"],
        "pick_count": 3,
        "description": "盈利+低波+反转+流动性，稳健均衡配置",
        "style": "均衡型",
    },
}


# ============================================================
# 查询接口
# ============================================================

def list_all_factors() -> list[FactorDefinition]:
    """返回所有因子定义"""
    return list(FACTOR_DEFINITIONS.values())


def list_factors_by_group(group: str) -> list[FactorDefinition]:
    """返回指定分组的所有因子"""
    names = FACTOR_GROUPS.get(group, [])
    return [FACTOR_DEFINITIONS[n] for n in names if n in FACTOR_DEFINITIONS]


def list_factors_by_source(source: FactorSource) -> list[FactorDefinition]:
    """返回指定数据源的所有因子"""
    return [f for f in FACTOR_DEFINITIONS.values() if f.source == source]


def get_factor(name: str) -> Optional[FactorDefinition]:
    """获取单个因子定义"""
    return FACTOR_DEFINITIONS.get(name)


def get_daily_factor_names() -> list[str]:
    """返回所有日线因子的列名"""
    return [f.name for f in FACTOR_DEFINITIONS.values() if f.source == FactorSource.DAILY]


def get_financial_factor_names() -> list[str]:
    """返回所有财务因子的列名"""
    return [f.name for f in FACTOR_DEFINITIONS.values() if f.source == FactorSource.FINANCIAL]


def get_fin_mapping() -> dict[str, str]:
    """返回 {fin_col: factor_name} 的映射"""
    return {f.fin_col: f.name for f in FACTOR_DEFINITIONS.values()
            if f.source == FactorSource.FINANCIAL and f.fin_col}
