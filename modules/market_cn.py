import streamlit as st
import akshare as ak
import pandas as pd
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("market_cn")


# =============================================================================
# 指数数据
# =============================================================================

@st.cache_data(ttl=300)
def fetch_cn_indices() -> pd.DataFrame:
    """获取 A 股主要指数"""
    try:
        logger.info("Fetching CN indices...")
        df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
        logger.info(f"Fetched CN indices: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error fetching CN indices: {e}", exc_info=True)
        st.error(f"获取指数数据失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_sz_indices() -> pd.DataFrame:
    """获取深证系列指数"""
    try:
        df = ak.stock_zh_index_spot_em(symbol="深证系列指数")
        return df
    except Exception as e:
        st.error(f"获取深证指数失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 板块热度
# =============================================================================

@st.cache_data(ttl=600)
def fetch_sector_heat() -> pd.DataFrame:
    """获取行业板块排名"""
    try:
        df = ak.stock_board_industry_name_em()
        return df.head(20)
    except Exception as e:
        st.error(f"获取板块数据失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 北向资金
# =============================================================================

@st.cache_data(ttl=600)
def fetch_north_funds() -> pd.DataFrame:
    """获取北向资金流向汇总"""
    try:
        logger.info("Fetching North Funds...")
        df = ak.stock_hsgt_fund_flow_summary_em()
        return df
    except Exception as e:
        logger.error(f"Error fetching North Funds: {e}", exc_info=True)
        st.error(f"获取北向资金失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_north_funds_hist() -> pd.DataFrame:
    """获取北向资金历史数据"""
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        return df.tail(30)
    except Exception as e:
        st.error(f"获取北向资金历史失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 个股查询
# =============================================================================

@st.cache_data(ttl=300)
def fetch_stock_data(code: str) -> pd.DataFrame:
    """获取个股日K数据"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        return df.tail(60)
    except Exception as e:
        st.error(f"获取股票数据失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 宏观数据
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_m2_supply() -> pd.DataFrame:
    """获取 M2 货币供应年率"""
    try:
        df = ak.macro_china_m2_yearly()
        df = df.dropna(subset=["今值"])
        return df.tail(24)
    except Exception as e:
        st.error(f"获取 M2 数据失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_unemployment() -> pd.DataFrame:
    """获取城镇调查失业率"""
    try:
        df = ak.macro_china_urban_unemployment()
        if "item" in df.columns:
            df = df[df["item"] == "全国城镇调查失业率"]
        return df.tail(24)
    except Exception as e:
        st.error(f"获取失业率数据失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_fx_reserves() -> pd.DataFrame:
    """获取外汇储备"""
    try:
        df = ak.macro_china_fx_reserves_yearly()
        df = df.dropna(subset=["今值"])
        return df.tail(24)
    except Exception as e:
        st.error(f"获取外汇储备失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200)
def fetch_enterprise_boom() -> pd.DataFrame:
    """获取企业景气指数"""
    try:
        df = ak.macro_china_enterprise_boom_index()
        return df.head(20)
    except Exception as e:
        st.error(f"获取企业景气指数失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200)
def fetch_macro_leverage() -> pd.DataFrame:
    """获取中国宏观杠杆率"""
    try:
        df = ak.macro_cnbs()
        return df.tail(20)
    except Exception as e:
        st.error(f"获取宏观杠杆率失败: {e}")
        return pd.DataFrame()
