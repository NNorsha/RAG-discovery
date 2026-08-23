#!/bin/bash
set -e
# Cloud Studio 一键启动脚本
# 运行前请先在 Cloud Studio「环境变量」或 .env 中设置 LLM_API_KEY

echo "==> 安装依赖..."
pip install -r requirements.txt

echo "==> 使用云端 Embedding 后端(省内存、免下载模型)..."
export EMBED_BACKEND=api
export EMBED_MODEL=embedding-3

# 若未在环境变量中设置,则使用 .env 中的值
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "==> 入库示例文档..."
python ingest.py

echo "==> 启动 Streamlit(端口 8080)..."
streamlit run app.py \
  --server.port 8080 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
