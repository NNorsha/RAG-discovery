"""个人知识库问答 —— Streamlit 界面。
<<<<<<< HEAD
### 终端---cd "J:\1AI工具\Codex1\RAG\RAG-discovery-main"
### streamlit run app.py       ctrl+C停止

=======

运行: streamlit run app.py
>>>>>>> e6b7f722f10d0a2104525ea5f0a2883dd39ae081
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import rag_core as rag

st.set_page_config(page_title="个人知识库问答", page_icon="📚", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 首次启动(如云端部署)知识库为空时,自动入库自带示例文档
if not rag.list_sources():
    sample_files = rag.scan_folder(DATA_DIR)
    if sample_files:
        with st.spinner("首次启动,正在入库示例文档…"):
            rag.ingest_files(sample_files)


def render_sources(contexts: list[dict]) -> None:
    with st.expander(f"参考来源({len(contexts)} 条)"):
        for i, c in enumerate(contexts):
            snippet = c["text"][:220] + ("…" if len(c["text"]) > 220 else "")
            st.markdown(
                f"**[{i + 1}]** `{c['source']}` · 片段 {c['chunk']} · 相似度 {c['score']}"
            )
            st.caption(snippet)


# ------------------------------------------------- 侧边栏:知识库管理

with st.sidebar:
    st.header("知识库管理")

    uploaded = st.file_uploader(
        "上传文档(md / txt / pdf / docx)",
        type=["md", "txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    folder = st.text_input("或扫描文件夹(相对项目目录)", value="data")

    if st.button("入库", type="primary", use_container_width=True):
        files: list[Path] = []
        for f in uploaded or []:
            target = DATA_DIR / f.name
            target.write_bytes(f.getbuffer())
            files.append(target)
        folder_path = Path(folder)
        if not folder_path.is_absolute():
            folder_path = Path(__file__).resolve().parent / folder_path
        if folder_path.is_dir():
            files.extend(rag.scan_folder(folder_path))
        files = sorted(set(files))
        if not files:
            st.warning("没有找到可入库的文档")
        else:
            bar = st.progress(0.0, text="准备中…")
            stats = rag.ingest_files(
                files,
                progress_cb=lambda done, total, name: bar.progress(
                    done / total, text=f"处理 {name}({done}/{total})"
                ),
            )
            bar.empty()
            st.success(f"入库完成:{stats['files']} 个文件,{stats['chunks']} 个分块")
            for s in stats["skipped"]:
                st.caption(f"跳过:{s}")
            st.rerun()

    st.divider()
    sources = rag.list_sources()
    st.subheader(f"已入库 {len(sources)} 个文档")
    for s in sources:
        st.markdown(f"- `{s['source']}` — {s['chunks']} 块")
    if sources and st.button("清空知识库", use_container_width=True):
        rag.clear()
        st.rerun()

    st.divider()
    top_k = st.slider("检索片段数 Top-K", 1, 8, 4)
    st.caption(f"向量模型:{rag.EMBED_MODEL}(本地)")
    if rag.llm_configured():
        st.caption("LLM:已配置")
    else:
        st.warning("未配置 LLM key,当前为仅检索模式。复制 .env.example 为 .env 并填入 key 开启问答。")

# ------------------------------------------------- 主区:问答

<<<<<<< HEAD
st.title("Norsha写书中...随便提问")
=======
st.title("个人知识库问答")
>>>>>>> e6b7f722f10d0a2104525ea5f0a2883dd39ae081
st.caption("基于 RAG:向量检索本地文档,LLM 依据资料生成带来源引用的回答")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if question := st.chat_input("向你的知识库提问…"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    contexts = rag.query(question, top_k=top_k)
    with st.chat_message("assistant"):
        if not contexts:
            answer = "知识库为空或没有检索到相关内容,请先在左侧入库文档。"
            st.markdown(answer)
        elif not rag.llm_configured():
            answer = "未配置 LLM,当前仅展示检索结果:"
            st.markdown(answer)
            render_sources(contexts)
        else:
            answer = st.write_stream(rag.stream_answer(question, contexts))
            render_sources(contexts)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": contexts}
    )
