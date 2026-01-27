import streamlit as st
import akshare as ak
import pandas as pd
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("futures")


# =============================================================================
# 贵金属
# =============================================================================

@st.cache_data(ttl=600)
def fetch_gold_price() -> pd.DataFrame:
    """获取黄金现货价格"""
    try:
        logger.info("Fetching Gold Price...")
        df = ak.spot_golden_benchmark_sge()
        return df.tail(30)
    except Exception as e:
        logger.error(f"Error fetching Gold Price: {e}", exc_info=True)
        st.error(f"获取黄金价格失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_silver_price() -> pd.DataFrame:
    """获取白银现货价格"""
    try:
        df = ak.spot_silver_sge()
        return df.tail(30)
    except Exception as e:
        st.error(f"获取白银价格失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 原油
# =============================================================================

@st.cache_data(ttl=600)
def fetch_crude_oil() -> pd.DataFrame:
    """获取原油期货价格"""
    try:
        df = ak.futures_foreign_commodity_realtime(symbol="CL")
        return df
    except Exception as e:
        st.error(f"获取原油期货失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 外盘期货汇总
# =============================================================================

@st.cache_data(ttl=300)
def fetch_foreign_futures() -> pd.DataFrame:
    """获取外盘期货实时行情"""
    try:
        # 获取主要外盘期货
        symbols = ["CL", "GC", "SI", "HG", "NG"]  # 原油、黄金、白银、铜、天然气
        futures_data = []
        
        for symbol in symbols:
            try:
                df = ak.futures_foreign_commodity_realtime(symbol=symbol)
                if not df.empty:
                    latest = df.iloc[0] if len(df) > 0 else {}
                    futures_data.append({
                        "代码": symbol,
                        "名称": get_futures_name(symbol),
                        "最新价": latest.get("current_price", "--"),
                        "涨跌幅": latest.get("change_percent", "--"),
                    })
            except Exception:
                pass
        
        return pd.DataFrame(futures_data)
    except Exception as e:
        st.error(f"获取外盘期货失败: {e}")
        return pd.DataFrame()


def get_futures_name(symbol: str) -> str:
    """获取期货代码对应名称"""
    names = {
        "CL": "WTI 原油",
        "GC": "COMEX 黄金",
        "SI": "COMEX 白银",
        "HG": "COMEX 铜",
        "NG": "天然气",
    }
    return names.get(symbol, symbol)


# =============================================================================
# 外汇
# =============================================================================

@st.cache_data(ttl=600)
def fetch_forex() -> pd.DataFrame:
    """获取主要外汇汇率"""
    try:
        df = ak.currency_boc_sina()
        # 筛选主要货币对
        major_currencies = ["美元", "欧元", "英镑", "日元", "港币"]
        if "货币名称" in df.columns:
            df = df[df["货币名称"].isin(major_currencies)]
        return df
    except Exception as e:
        st.error(f"获取外汇汇率失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 国内商品期货
# =============================================================================

@st.cache_data(ttl=600)
def fetch_cn_futures() -> pd.DataFrame:
    """获取国内商品期货主力合约"""
    try:
        df = ak.futures_main_sina()
        return df.head(20)
    except Exception as e:
        st.error(f"获取国内期货失败: {e}")
        return pd.DataFrame()
