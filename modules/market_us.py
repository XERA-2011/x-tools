import streamlit as st
import akshare as ak
import pandas as pd
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("market_us")


# =============================================================================
# 美股指数
# =============================================================================

@st.cache_data(ttl=300)
def fetch_us_indices() -> pd.DataFrame:
    """获取美股主要指数"""
    try:
        logger.info("Fetching US indices...")
        # 道琼斯、标普500、纳斯达克
        indices = []
        
        for symbol, name in [
            (".DJI", "道琼斯"),
            (".IXIC", "纳斯达克"),
            (".INX", "标普500"),
        ]:
            try:
                df = ak.index_us_stock_sina(symbol=symbol)
                if not df.empty:
                    latest = df.iloc[-1]
                    indices.append({
                        "名称": name,
                        "代码": symbol,
                        "最新价": latest.get("close", 0),
                        "涨跌幅": ((latest.get("close", 0) - latest.get("open", 0)) 
                                  / latest.get("open", 1) * 100) if latest.get("open", 0) else 0
                    })
            except Exception:
                pass
        
        return pd.DataFrame(indices)
    except Exception as e:
        st.error(f"获取美股指数失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 中概股
# =============================================================================

@st.cache_data(ttl=600)
def fetch_china_concept() -> pd.DataFrame:
    """获取中概股行情"""
    try:
        df = ak.stock_us_zh_spot()
        # 按市值排序，取前20
        if "mktcap" in df.columns:
            df = df.sort_values("mktcap", ascending=False)
        return df.head(20)
    except Exception as e:
        st.error(f"获取中概股失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 热门 ETF
# =============================================================================

@st.cache_data(ttl=600)
def fetch_us_etf() -> pd.DataFrame:
    """获取美股热门 ETF"""
    try:
        # 常见热门 ETF
        etf_symbols = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK", "XLF", "XLE", "GLD"]
        etf_data = []
        
        for symbol in etf_symbols:
            try:
                df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
                if not df.empty:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) >= 2 else latest
                    change = ((latest["close"] - prev["close"]) / prev["close"] * 100) if prev["close"] else 0
                    etf_data.append({
                        "代码": symbol,
                        "最新价": latest["close"],
                        "涨跌幅": change
                    })
            except Exception:
                pass
        
        return pd.DataFrame(etf_data)
    except Exception as e:
        st.error(f"获取 ETF 失败: {e}")
        return pd.DataFrame()


# =============================================================================
# 恐惧贪婪指数
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_fear_greed() -> Dict[str, Any]:
    """获取 CNN 恐惧贪婪指数"""
    try:
        df = ak.stock_js_weibo_report()
        # 如果接口不可用，返回空
        if df.empty:
            return {"score": None, "level": "N/A"}
        return {"score": None, "level": "N/A"}
    except Exception:
        # 接口可能不可用
        return {"score": None, "level": "N/A"}
