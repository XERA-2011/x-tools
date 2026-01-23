"""
x-streamlit: 股票数据分析工具
基于 Streamlit + AkShare 构建
"""

import os
import streamlit as st
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

# 页面配置
st.set_page_config(
    page_title="x-streamlit 数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .stMetric > div {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 密码保护
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



def main():
    st.title("📊 x-streamlit 数据分析")
    st.caption(f"数据来源: AkShare | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 侧边栏
    with st.sidebar:
        st.header("🔧 工具箱")
        tool = st.radio(
            "选择分析工具",
            ["🏠 首页", "📈 个股查询", "🔥 板块热度", "💰 北向资金"]
        )

    # 路由
    if tool == "🏠 首页":
        show_home()
    elif tool == "📈 个股查询":
        show_stock_query()
    elif tool == "🔥 板块热度":
        show_sector_heat()
    elif tool == "💰 北向资金":
        show_north_funds()


@st.cache_data(ttl=300)
def fetch_index_data() -> pd.DataFrame:
    """获取指数数据"""
    try:
        df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
        return df
    except Exception as e:
        st.error(f"获取指数数据失败: {e}")
        return pd.DataFrame()


def show_home():
    """首页概览"""
    st.subheader("🌍 全球指数")
    
    with st.spinner("正在获取指数数据..."):
        df = fetch_index_data()
    
    if not df.empty:
        # 筛选主要指数
        key_indices = ["上证指数", "深证成指", "创业板指", "科创50", "沪深300", "中证500"]
        df_main = df[df["名称"].isin(key_indices)]
        
        # 显示主要指数卡片
        cols = st.columns(3)
        for i, (_, row) in enumerate(df_main.iterrows()):
            with cols[i % 3]:
                change = row.get("涨跌幅", 0)
                change_str = f"{change:+.2f}%" if pd.notna(change) else "N/A"
                price = row.get("最新价", 0)
                price_str = f"{price:,.2f}" if pd.notna(price) else "N/A"
                st.metric(
                    label=row["名称"],
                    value=price_str,
                    delta=change_str
                )
        
        # 显示完整指数表格
        with st.expander("📊 查看全部指数", expanded=False):
            display_cols = ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[available_cols].head(30),
                use_container_width=True,
                hide_index=True
            )
    
    st.info("💡 提示: 使用左侧菜单选择分析工具")


@st.cache_data(ttl=300)
def fetch_stock_data(code: str) -> pd.DataFrame:
    """获取股票数据（带缓存）"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        return df.tail(60)  # 最近60天
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return pd.DataFrame()


def show_stock_query():
    """个股查询"""
    st.subheader("📈 个股查询")
    
    code = st.text_input("输入股票代码", placeholder="例如: 000001")
    
    if code:
        with st.spinner("正在获取数据..."):
            df = fetch_stock_data(code)
        
        if not df.empty:
            # 显示最新价格
            latest = df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("最新价", f"¥{latest['收盘']:.2f}")
            col2.metric("成交量", f"{latest['成交量']/10000:.0f}万手")
            col3.metric("成交额", f"{latest['成交额']/100000000:.2f}亿")
            
            # K线图
            st.line_chart(df.set_index("日期")["收盘"])
            
            # 数据表格
            with st.expander("查看详细数据"):
                st.dataframe(df, use_container_width=True)


@st.cache_data(ttl=600)
def fetch_sector_data() -> pd.DataFrame:
    """获取板块数据"""
    try:
        df = ak.stock_board_industry_name_em()
        return df.head(20)
    except Exception as e:
        st.error(f"获取板块数据失败: {e}")
        return pd.DataFrame()


def show_sector_heat():
    """板块热度"""
    st.subheader("🔥 板块热度 Top 20")
    
    with st.spinner("正在获取板块数据..."):
        df = fetch_sector_data()
    
    if not df.empty:
        # 显示表格
        st.dataframe(
            df[["板块名称", "最新价", "涨跌幅", "总市值"]],
            use_container_width=True,
            hide_index=True
        )


@st.cache_data(ttl=600)
def fetch_north_funds() -> pd.DataFrame:
    """获取北向资金数据"""
    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        return df
    except Exception as e:
        st.error(f"获取北向资金失败: {e}")
        return pd.DataFrame()


def show_north_funds():
    """北向资金"""
    st.subheader("💰 北向资金流向")
    
    with st.spinner("正在获取北向资金数据..."):
        df = fetch_north_funds()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    if check_password():
        main()

