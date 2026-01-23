# x-streamlit

基于 Streamlit + AkShare 的股票数据分析工具。

## 功能

- 📈 **个股查询** - 输入股票代码查看价格走势
- 🔥 **板块热度** - 查看行业板块涨跌排行
- 💰 **北向资金** - 查看沪深港通资金流向

## 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py

# 访问 http://localhost:8501
```

## Docker 构建

```bash
# 构建镜像
docker build -t x-streamlit .

# 运行容器
docker run -p 8501:8501 x-streamlit

# 访问 http://localhost:8501/streamlit
```

## 部署

推送代码后，GitHub Actions 会自动：
1. 构建 Docker 镜像
2. 推送到阿里云容器镜像服务 (ACR)

服务器通过 `x-actions` 仓库的配置拉取镜像并启动。

**线上访问**：`http://你的域名/streamlit/`

## 技术栈

- [Streamlit](https://streamlit.io/) - Python Web 框架
- [AkShare](https://akshare.akfamily.xyz/) - 金融数据接口
- [Pandas](https://pandas.pydata.org/) - 数据处理
