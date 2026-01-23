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
            ["🏠 首页", "📊 宏观数据", "📈 个股查询", "🔥 板块热度", "💰 北向资金"]
        )

    # 路由
    if tool == "🏠 首页":
        show_home()
    elif tool == "📊 宏观数据":
        show_macro_data()
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


# =============================================================================
# 宏观数据模块
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
        # 数据格式: date, item, value
        # 只取 "全国城镇调查失业率" 这一项
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
        st.error(f"获取外汇储备数据失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200)
def fetch_enterprise_boom() -> pd.DataFrame:
    """获取企业景气 & 企业家信心指数"""
    try:
        df = ak.macro_china_enterprise_boom_index()
        # API 返回的是倒序（最新在前），我们取前 20 条即可
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


def show_macro_data():
    """宏观经济数据展示"""
    st.subheader("📊 中国宏观经济数据")
    st.caption("数据来源: 东方财富、国家统计局、中国人民银行")
    
    # 加载数据
    with st.spinner("正在获取宏观数据..."):
        m2_df = fetch_m2_supply()
        unemployment_df = fetch_unemployment()
        fx_df = fetch_fx_reserves()
        boom_df = fetch_enterprise_boom()
        leverage_df = fetch_macro_leverage()
    
    # =========================================================================
    # 核心指标卡片
    # =========================================================================
    st.markdown("### 💹 核心指标")
    
    col1, col2, col3 = st.columns(3)
    
    # M2 货币供应
    with col1:
        if not m2_df.empty:
            latest = m2_df.iloc[-1]
            prev = m2_df.iloc[-2] if len(m2_df) >= 2 else latest
            value = latest.get("今值", 0)
            prev_value = prev.get("今值", 0)
            date_str = str(latest.get("日期", ""))[:10]
            delta = value - prev_value if pd.notna(value) and pd.notna(prev_value) else 0
            st.metric(
                label="M2 货币供应年率",
                value=f"{value:.1f}%" if pd.notna(value) else "N/A",
                delta=f"{delta:+.1f}%" if delta != 0 else "持平",
                delta_color="normal"
            )
            if date_str:
                st.caption(f"📅 {date_str}")
        else:
            st.metric("M2 货币供应年率", "加载失败")
    
    # 城镇失业率
    with col2:
        if not unemployment_df.empty:
            latest = unemployment_df.iloc[-1]
            prev = unemployment_df.iloc[-2] if len(unemployment_df) >= 2 else latest
            # 数据格式: date, item, value
            if "value" in unemployment_df.columns:
                value = latest.get("value", 0)
                prev_value = prev.get("value", 0)
                date_str = str(latest.get("date", ""))
                # 格式化日期 202512 -> 2025-12
                if len(date_str) == 6:
                    date_str = f"{date_str[:4]}-{date_str[4:]}"
            else:
                # 兼容旧格式
                value_col = [c for c in unemployment_df.columns if "失业率" in c]
                if value_col:
                    value = latest.get(value_col[0], 0)
                    prev_value = prev.get(value_col[0], 0)
                else:
                    value = prev_value = 0
                date_str = ""
            
            delta = value - prev_value if pd.notna(value) and pd.notna(prev_value) else 0
            st.metric(
                label="城镇调查失业率",
                value=f"{value:.1f}%" if pd.notna(value) else "N/A",
                delta=f"{delta:+.1f}%" if delta != 0 else "持平",
                delta_color="inverse"  # 失业率下降是好事
            )
            if date_str:
                st.caption(f"📅 {date_str}")
        else:
            st.metric("城镇调查失业率", "加载失败")
    
    # 外汇储备
    with col3:
        if not fx_df.empty:
            latest = fx_df.iloc[-1]
            prev = fx_df.iloc[-2] if len(fx_df) >= 2 else latest
            value = latest.get("今值", 0)
            prev_value = prev.get("今值", 0)
            date_str = str(latest.get("日期", ""))[:10]
            delta = value - prev_value if pd.notna(value) and pd.notna(prev_value) else 0
            st.metric(
                label="外汇储备 (亿美元)",
                value=f"{value:,.0f}" if pd.notna(value) else "N/A",
                delta=f"{delta:+,.0f}" if delta != 0 else "持平"
            )
            if date_str:
                st.caption(f"📅 {date_str}")
        else:
            st.metric("外汇储备", "加载失败")
    
    st.divider()
    
    # =========================================================================
    # M2 趋势图
    # =========================================================================
    st.markdown("### 📈 M2 货币供应趋势")
    if not m2_df.empty and "今值" in m2_df.columns:
        chart_df = m2_df[["日期", "今值"]].copy()
        chart_df["日期"] = pd.to_datetime(chart_df["日期"])
        chart_df = chart_df.set_index("日期")
        chart_df.columns = ["M2年率(%)"]
        st.line_chart(chart_df, use_container_width=True)
    else:
        st.info("暂无 M2 数据可显示")
    
    st.divider()
    
    # =========================================================================
    # 企业景气指数
    # =========================================================================
    st.markdown("### 💼 企业景气指数")
    if not boom_df.empty:
        # 查找最新的有效数据（非 NaN）
        latest_valid_idx = -1
        for i in range(len(boom_df)):
            if pd.notna(boom_df.iloc[i].get("企业景气指数-指数")):
                latest_valid_idx = i
                break
        
        latest_row = boom_df.iloc[latest_valid_idx] if latest_valid_idx != -1 else boom_df.iloc[0]
        latest_quarter = latest_row.get("季度", "")

        # 显示双指标
        col1, col2 = st.columns(2)
        
        with col1:
            if "企业景气指数-指数" in boom_df.columns:
                val = latest_row.get("企业景气指数-指数")
                st.metric("企业景气指数", f"{val:.1f}" if pd.notna(val) else "N/A", help=f"数据季度: {latest_quarter}")
        
        with col2:
            if "企业家信心指数-指数" in boom_df.columns:
                # 尝试找企业家信心的最新有效值
                conf_val = latest_row.get("企业家信心指数-指数")
                # 如果当前行是 NaN，往后找找有没有
                if pd.isna(conf_val):
                    for i in range(len(boom_df)):
                         v = boom_df.iloc[i].get("企业家信心指数-指数")
                         if pd.notna(v):
                             conf_val = v
                             break
                st.metric("企业家信心指数", f"{conf_val:.1f}" if pd.notna(conf_val) else "N/A")
        
        # 趋势图 (反转顺序，时间升序)
        with st.expander("📊 查看趋势图", expanded=True):
            chart_cols = []
            if "企业景气指数-指数" in boom_df.columns:
                chart_cols.append("企业景气指数-指数")
            if "企业家信心指数-指数" in boom_df.columns:
                chart_cols.append("企业家信心指数-指数")
            
            if chart_cols and "季度" in boom_df.columns:
                chart_df = boom_df[["季度"] + chart_cols].copy()
                # 反转顺序用于绘图
                chart_df = chart_df.iloc[::-1]
                chart_df = chart_df.set_index("季度")
                st.line_chart(chart_df, use_container_width=True)
    else:
        st.info("暂无企业景气数据")
    
    st.divider()
    
    # =========================================================================
    # 宏观杠杆率
    # =========================================================================
    st.markdown("### 🏛️ 中国宏观杠杆率")
    if not leverage_df.empty:
        # 最新值展示
        latest = leverage_df.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            val = latest.get("居民部门", 0)
            st.metric("居民部门", f"{val:.1f}%" if pd.notna(val) else "N/A")
        with col2:
            val = latest.get("非金融企业部门", 0)
            st.metric("非金融企业", f"{val:.1f}%" if pd.notna(val) else "N/A")
        with col3:
            val = latest.get("政府部门", 0)
            st.metric("政府部门", f"{val:.1f}%" if pd.notna(val) else "N/A")
        with col4:
            val = latest.get("实体经济部门", 0)
            st.metric("实体经济合计", f"{val:.1f}%" if pd.notna(val) else "N/A")
        
        # 趋势图 - 堆叠面积图
        with st.expander("📊 查看杠杆率趋势", expanded=True):
            plot_cols = ["居民部门", "非金融企业部门", "政府部门"]
            available_cols = [c for c in plot_cols if c in leverage_df.columns]
            
            if available_cols and "年份" in leverage_df.columns:
                chart_df = leverage_df[["年份"] + available_cols].copy()
                chart_df = chart_df.set_index("年份")
                st.area_chart(chart_df, use_container_width=True)
        
        # 详细数据表格
        with st.expander("📋 查看详细数据"):
            st.dataframe(leverage_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无宏观杠杆率数据")


if __name__ == "__main__":
    if check_password():
        main()

