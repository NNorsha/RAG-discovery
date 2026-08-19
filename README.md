# RAG 个人知识库问答

基于 RAG(检索增强生成)架构的本地知识库问答系统:上传文档后用自然语言提问,
系统检索相关片段,由大模型生成**带来源引用**的回答,引用可逐条点开核对。

## 架构

```
文档(md/txt/pdf/docx)
  -> 加载解析 -> 段落感知分块(500 字,80 字重叠)
  -> BGE-small-zh 本地向量化 -> ChromaDB 持久化(余弦 / HNSW)
提问 -> 问题向量化 -> Top-K 检索 -> 拼入 Prompt -> LLM 流式生成(标注引用编号)
```

- 向量模型:`BAAI/bge-small-zh-v1.5`,本地运行,零 API 成本,私有数据不出门
- 向量库:ChromaDB,本地持久化
- LLM:任意 OpenAI 兼容接口,默认智谱 `glm-4-flash`(免费)
- 界面:Streamlit,流式输出 + 引用展开

## 快速开始

```powershell
pip install -r requirements.txt
copy .env.example .env     # 然后编辑 .env 填入 LLM key
python ingest.py           # 入库 ./data 下的文档(也可在网页里上传)
streamlit run app.py       # 打开 http://localhost:8501
python selfcheck.py        # 可选:检索冒烟测试
```

未配置 LLM 也能运行:界面退化为"仅检索"模式,方便单独验证召回质量。

### 无法访问 HuggingFace?

向量模型默认由 fastembed 从 HuggingFace 自动下载。网络不通时走 ModelScope 镜像:

1. 从 <https://modelscope.cn/models/BAAI/bge-small-zh-v1.5/files> 下载
   `config.json` `tokenizer.json` `tokenizer_config.json` `special_tokens_map.json`
   `vocab.txt` `model.safetensors` 放入 `models/bge-small-zh/`
2. `pip install torch transformers` 后运行 `python export_onnx.py`,
   本地导出 ONNX 并自动做精度校验(与 PyTorch 输出余弦相似度 ≥ 0.9999)
3. 之后 `rag_core` 会优先加载 `models/bge-small-zh/`,完全离线

## 目录结构

```
rag_core.py      加载/分块/向量化/检索/生成的核心逻辑
app.py           Streamlit 界面(知识库管理 + 流式问答 + 引用展开)
ingest.py        命令行批量入库
selfcheck.py     检索冒烟测试
export_onnx.py   (可选)离线环境的 ONNX 导出工具
data/notes/      示例知识库(三篇 RAG/Embedding/Prompt 笔记)
```

## 隐私设计

- `.env`(API key)、向量库 `.chroma/`、模型文件 `models/` 均已 gitignore
- `data/` 下除示例外的新增文档默认不进 git,私人文档放心放
- Embedding 全程本地推理,文档内容不会发给任何第三方;
  仅提问时检索到的相关片段会随请求发给配置的 LLM 服务

## 在线部署(让任何人通过链接访问)

本项目可直接部署到 Streamlit Cloud(免费):

1. Fork 本仓库到你自己的 GitHub 账号
2. 打开 [share.streamlit.io](https://share.streamlit.io),用 GitHub 登录
3. New app → 选择该仓库 → Main file path 填 `app.py` → Deploy
4. 部署前在 Advanced settings → Secrets 中填入(对应 `.env` 的同名变量):

```toml
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_API_KEY = "你的 API key"
LLM_MODEL = "glm-4-flash"
```

部署完成后会得到一个 `https://<app-name>.streamlit.app` 公开链接,任何人可访问。
注意:公开部署意味着所有访客共用你的 LLM 额度,建议使用免费/限额模型,
并留意知识库示例文档不要包含私人信息。

## 简历话术

> 基于 RAG 架构独立开发个人知识库问答系统:实现 md/pdf/docx 多格式文档解析、
> 段落感知分块与重叠策略;使用本地化 BGE Embedding 模型与 ChromaDB 向量库
> 完成语义检索,Embedding 侧零 API 成本;设计带来源编号引用的 Prompt 模板
> 抑制幻觉并支持溯源核对;通过 Streamlit 提供流式问答与知识库管理界面。

EN: Built a RAG-based personal knowledge-base QA system: multi-format document
parsing, paragraph-aware chunking with overlap, local BGE embeddings with
ChromaDB vector search (zero embedding API cost), citation-grounded prompting
to reduce hallucination, and a streaming Streamlit UI.

## 面试可讲的技术取舍

- 为什么本地 Embedding:私有数据不出本机、零边际成本、可离线运行
- 为什么段落分块 + 重叠:避免关键句被硬切截断,保住语义完整度
- 为什么要求引用编号:幻觉可核对,这正是 RAG 相对裸调模型的核心价值
- 已知边界与下一步:rerank 精排、BM25+向量混合检索、固定评测集回归
