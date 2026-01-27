"""
x-streamlit: 股票数据分析工具
基于 Streamlit + AkShare 构建

三大板块: 中国市场 | 美国市场 | 全球期货
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional
import requests
import functools
import logging
import sys

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

# =============================================================================
# Patch Requests for AkShare
# =============================================================================
# 修复 AkShare 在 Docker 中因 User-Agent 被封的问题
_original_session_request = requests.Session.request

@functools.wraps(_original_session_request)
def _patched_request(self, method, url, *args, **kwargs):
    headers = kwargs.get("headers", {})
    if not headers:
        headers = {}
    
    # 强制注入浏览器 UA
    if "User-Agent" not in headers or "python" in headers["User-Agent"].lower():
        headers["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # 禁用 Keep-Alive (解决 RemoteDisconnected)
    headers["Connection"] = "close"
    
    # 强制 HTTP (解决 Docker SSL 问题)
    if "push2.eastmoney.com" in url and url.startswith("https://"):
        url = url.replace("https://", "http://")
        
    kwargs["headers"] = headers
    
    # Debug Logging
    # logger.info(f"Request: {method} {url}")
    # logger.info(f"Headers: {headers}")
    
    try:
        response = _original_session_request(self, method, url, *args, **kwargs)
        # logger.info(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {method} {url} - {e}")
        raise e

requests.Session.request = _patched_request


# 导入模块
from modules import market_cn, market_us, futures

# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="x-streamlit 数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义样式
st.markdown("""
<style>
    /* 顶部 Tab 样式优化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0e1117;
        padding: 8px 16px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
    }
    /* 指标卡片样式 */
    .stMetric > div {
        background-color: #1e1e1e;
        padding: 12px;
        border-radius: 8px;
    }
    /* 紧凑模式 */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 密码保护
# =============================================================================

CORRECT_PASSWORD = os.environ.get("STREAMLIT_PASSWORD", "xera2011")


def check_password() -> bool:
    """密码验证"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 请输入访问密码")
    password = st.text_input("密码", type="password", key="password_input")
    
    if st.button("登录", type="primary"):
        if password == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误，请重试")
    
    return False


# =============================================================================
# 中国市场板块
# =============================================================================

def show_cn_market():
    """中国市场"""
    st.subheader("🇨🇳 中国市场")
    st.caption(f"数据来源: AkShare | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 子 Tab
    cn_tab1, cn_tab2, cn_tab3, cn_tab4 = st.tabs(["📈 主要指数", "🔥 板块热度", "💰 北向资金", "📊 宏观数据"])
    
    with cn_tab1:
        show_cn_indices()
    
    with cn_tab2:
        show_cn_sectors()
    
    with cn_tab3:
        show_north_funds()
    
    with cn_tab4:
        show_cn_macro()


def show_cn_indices():
    """中国主要指数"""
    with st.spinner("正在获取指数数据..."):
        df = market_cn.fetch_cn_indices()
    
    if not df.empty:
        # 筛选主要指数
        key_indices = ["上证指数", "深证成指", "创业板指", "科创50", "沪深300", "中证500"]
        df_main = df[df["名称"].isin(key_indices)]
        
        # 显示指数卡片
        cols = st.columns(3)
        for i, (_, row) in enumerate(df_main.iterrows()):
            with cols[i % 3]:
                change = row.get("涨跌幅", 0)
                change_str = f"{change:+.2f}%" if pd.notna(change) else "--"
                price = row.get("最新价", 0)
                price_str = f"{price:,.2f}" if pd.notna(price) else "--"
                st.metric(
                    label=row["名称"],
                    value=price_str,
                    delta=change_str
                )
        
        # 详细表格
        with st.expander("📊 查看全部指数"):
            display_cols = ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交量", "成交额"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols].head(30), width="stretch", hide_index=True)


def show_cn_sectors():
    """板块热度"""
    with st.spinner("正在获取板块数据..."):
        df = market_cn.fetch_sector_heat()
    
    if not df.empty:
        display_cols = ["板块名称", "最新价", "涨跌幅", "总市值"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], width="stretch", hide_index=True)


def show_north_funds():
    """北向资金"""
    with st.spinner("正在获取北向资金数据..."):
        df = market_cn.fetch_north_funds()
    
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
    
    # 历史趋势
    with st.expander("📈 北向资金历史"):
        hist_df = market_cn.fetch_north_funds_hist()
        if not hist_df.empty and "净买额" in hist_df.columns and "日期" in hist_df.columns:
            chart_df = hist_df[["日期", "净买额"]].copy()
            chart_df["日期"] = pd.to_datetime(chart_df["日期"])
            chart_df = chart_df.set_index("日期")
            st.line_chart(chart_df)


def show_cn_macro():
    """宏观数据"""
    with st.spinner("正在获取宏观数据..."):
        m2_df = market_cn.fetch_m2_supply()
        unemployment_df = market_cn.fetch_unemployment()
        fx_df = market_cn.fetch_fx_reserves()
    
    # 核心指标
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not m2_df.empty:
            latest = m2_df.iloc[-1]
            value = latest.get("今值", 0)
            date_str = str(latest.get("日期", ""))[:10]
            st.metric("M2 货币供应年率", f"{value:.1f}%" if pd.notna(value) else "--")
            if date_str:
                st.caption(f"📅 {date_str}")
        else:
            st.metric("M2 货币供应年率", "--")
    
    with col2:
        if not unemployment_df.empty:
            latest = unemployment_df.iloc[-1]
            if "value" in unemployment_df.columns:
                value = latest.get("value", 0)
            else:
                value = 0
            st.metric("城镇调查失业率", f"{value:.1f}%" if pd.notna(value) else "--")
        else:
            st.metric("城镇调查失业率", "--")
    
    with col3:
        if not fx_df.empty:
            latest = fx_df.iloc[-1]
            value = latest.get("今值", 0)
            st.metric("外汇储备 (亿美元)", f"{value:,.0f}" if pd.notna(value) else "--")
        else:
            st.metric("外汇储备", "--")
    
    # M2 趋势
    with st.expander("📈 M2 趋势"):
        if not m2_df.empty and "今值" in m2_df.columns:
            chart_df = m2_df[["日期", "今值"]].copy()
            chart_df["日期"] = pd.to_datetime(chart_df["日期"])
            chart_df = chart_df.set_index("日期")
            chart_df.columns = ["M2年率(%)"]
            st.line_chart(chart_df, width="stretch")


# =============================================================================
# 美国市场板块
# =============================================================================

def show_us_market():
    """美国市场"""
    st.subheader("🇺🇸 美国市场")
    st.caption(f"数据来源: AkShare | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    us_tab1, us_tab2, us_tab3 = st.tabs(["📈 主要指数", "🏮 中概股", "📊 热门 ETF"])
    
    with us_tab1:
        show_us_indices()
    
    with us_tab2:
        show_china_concept()
    
    with us_tab3:
        show_us_etf()


def show_us_indices():
    """美股主要指数"""
    with st.spinner("正在获取美股指数..."):
        df = market_us.fetch_us_indices()
    
    if not df.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(df.iterrows()):
            with cols[i % 3]:
                change = row.get("涨跌幅", 0)
                # 美股: 绿涨红跌
                delta_color = "normal" if change >= 0 else "inverse"
                st.metric(
                    label=row.get("名称", "--"),
                    value=f"{row.get('最新价', 0):,.2f}",
                    delta=f"{change:+.2f}%"
                )
    else:
        st.info("暂无美股指数数据，可能需要在美股交易时段获取")


def show_china_concept():
    """中概股"""
    with st.spinner("正在获取中概股数据..."):
        df = market_us.fetch_china_concept()
    
    if not df.empty:
        # 显示前20只中概股
        display_cols = ["名称", "最新价", "涨跌幅", "成交量"]
        available_cols = [c for c in display_cols if c in df.columns]
        if available_cols:
            st.dataframe(df[available_cols].head(20), width="stretch", hide_index=True)
        else:
            st.dataframe(df.head(20), width="stretch", hide_index=True)
    else:
        st.info("暂无中概股数据")


def show_us_etf():
    """热门 ETF"""
    with st.spinner("正在获取 ETF 数据..."):
        df = market_us.fetch_us_etf()
    
    if not df.empty:
        cols = st.columns(5)
        for i, (_, row) in enumerate(df.iterrows()):
            with cols[i % 5]:
                st.metric(
                    label=row.get("代码", "--"),
                    value=f"${row.get('最新价', 0):.2f}",
                    delta=f"{row.get('涨跌幅', 0):+.2f}%"
                )
    else:
        st.info("暂无 ETF 数据")


# =============================================================================
# 全球期货板块
# =============================================================================

def show_global_futures():
    """全球期货"""
    st.subheader("🌍 全球期货")
    st.caption(f"数据来源: AkShare | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    ft_tab1, ft_tab2, ft_tab3, ft_tab4 = st.tabs(["💰 贵金属", "🛢️ 能源", "💱 外汇", "🌾 国内期货"])
    
    with ft_tab1:
        show_metals()
    
    with ft_tab2:
        show_energy()
    
    with ft_tab3:
        show_forex()
    
    with ft_tab4:
        show_cn_futures()


def show_metals():
    """贵金属"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🥇 黄金")
        with st.spinner("获取黄金价格..."):
            gold_df = futures.fetch_gold_price()
        if not gold_df.empty:
            st.dataframe(gold_df.tail(10), width="stretch", hide_index=True)
        else:
            st.info("暂无黄金数据")
    
    with col2:
        st.markdown("##### 🥈 白银")
        with st.spinner("获取白银价格..."):
            silver_df = futures.fetch_silver_price()
        if not silver_df.empty:
            st.dataframe(silver_df.tail(10), width="stretch", hide_index=True)
        else:
            st.info("暂无白银数据")


def show_energy():
    """能源期货"""
    st.markdown("##### 🛢️ 外盘期货")
    with st.spinner("获取外盘期货..."):
        df = futures.fetch_foreign_futures()
    
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("暂无外盘期货数据")


def show_forex():
    """外汇"""
    st.markdown("##### 💱 主要汇率")
    with st.spinner("获取汇率数据..."):
        df = futures.fetch_forex()
    
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("暂无汇率数据")


def show_cn_futures():
    """国内期货"""
    st.markdown("##### 🌾 国内商品期货主力合约")
    with st.spinner("获取国内期货..."):
        df = futures.fetch_cn_futures()
    
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("暂无国内期货数据")


# =============================================================================
# 主程序
# =============================================================================

def main():
    st.title("📊 x-streamlit 数据分析")
    
    # 顶部 Tab 切换 - 三大板块
    tab_cn, tab_us, tab_futures = st.tabs(["🇨🇳 中国市场", "🇺🇸 美国市场", "🌍 全球期货"])
    
    with tab_cn:
        show_cn_market()
    
    with tab_us:
        show_us_market()
    
    with tab_futures:
        show_global_futures()


if __name__ == "__main__":
    if check_password():
        main()
