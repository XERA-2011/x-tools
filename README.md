# x-streamlit

基于 Streamlit + AkShare 的股票数据分析工具，包含 **中国市场**、**美国市场**、**全球期货** 三大板块。

## 功能

- 🇨🇳 **中国市场** - 主要指数、板块热度 Top 20、北向资金流向、宏观经济数据
- 🇺🇸 **美国市场** - 美股指数、中概股行情、热门 ETF
- 🌍 **全球期货** - 贵金属（金/银）、能源（原油/天然气）、外汇汇率、国内期货

## 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py

# 访问 http://localhost:8501
# 默认密码: xera2011
```

## Docker 部署

```bash
# 构建镜像
docker build -t x-streamlit .

# 运行容器
docker run -d -p 8501:8501 --name x-streamlit --rm x-streamlit

# 访问 http://localhost:8501
# (无需加 /streamlit 后缀)

# 查看日志
docker logs -f x-streamlit

# 停止容器
docker stop x-streamlit
```

## 项目结构

- `app.py`: 主程序入口（顶部 Tab 导航）
- `modules/`: 数据获取模块
  - `market_cn.py`: 中国市场
  - `market_us.py`: 美国市场
  - `futures.py`: 全球期货
- `.agent/`: 开发规范和工作流

## 技术栈

- [Streamlit](https://streamlit.io/) - Python Web 框架
- [AkShare](https://akshare.akfamily.xyz/) - 金融数据接口
- [Pandas](https://pandas.pydata.org/) - 数据处理
