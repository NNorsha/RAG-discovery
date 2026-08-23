"""RAG 核心模块:文档加载 -> 分块 -> 向量化 -> 检索 -> 生成。

向量模型: 本地 BGE-small-zh(EMBED_BACKEND=local,零 API 成本)
          或 OpenAI 兼容 Embedding 接口(EMBED_BACKEND=api,云端部署省内存)
向量库:   ChromaDB(本地持久化,余弦相似度 / HNSW)
LLM:     任意 OpenAI 兼容接口,通过 .env 配置
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

# HuggingFace 国内镜像(海外环境可删除本行);关闭 tokenizers 并行告警
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud 部署时从 st.secrets 读取配置(本地 .env 优先)
try:
    import streamlit as st

    for _k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBED_BACKEND",
               "EMBED_MODEL", "EMBED_BASE_URL", "EMBED_API_KEY"):
        if not os.getenv(_k) and _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / ".chroma"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local")  # local | api
# 两种后端的向量维度不同,用不同 collection 隔离,避免混库
COLLECTION_NAME = "knowledge_base" if EMBED_BACKEND == "local" else "knowledge_base_api"
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "BAAI/bge-small-zh-v1.5" if EMBED_BACKEND == "local" else "embedding-3",
)
# 本地 .env 里若残留 HF 风格模型名,API 模式下强制用服务端模型
if EMBED_BACKEND == "api" and "/" in EMBED_MODEL:
    EMBED_MODEL = "embedding-3"
SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".docx"}

# 本地模型目录(无 HF 网络时用 export_onnx.py 生成,见 README)
LOCAL_MODEL_DIR = BASE_DIR / "models" / "bge-small-zh"

# ---------------------------------------------------------------- 向量模型

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        kwargs = {}
        if (LOCAL_MODEL_DIR / "model_optimized.onnx").exists():
            kwargs["specific_model_path"] = str(LOCAL_MODEL_DIR)
        _embedder = TextEmbedding(model_name=EMBED_MODEL, **kwargs)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    if EMBED_BACKEND == "api":
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("EMBED_BASE_URL", os.getenv("LLM_BASE_URL")),
            api_key=os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY")),
        )
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    return [v.tolist() for v in _get_embedder().embed(texts)]


# ---------------------------------------------------------------- 文档加载

def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        for enc in ("utf-8", "gbk"):  # 兼容 Windows 中文存档
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx

        return "\n".join(
            p.text for p in docx.Document(str(path)).paragraphs if p.text.strip()
        )
    raise ValueError(f"不支持的文件类型: {ext}")


# ---------------------------------------------------------------- 分块

def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """按段落聚合切分,块间保留重叠,避免关键句在边界被切断。"""
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 1 > size:
            chunks.append(buf)
            buf = ""
        while len(para) > size:  # 超长段落硬切
            chunks.append(para[:size])
            para = para[size - overlap:]
        buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    if overlap > 0:
        chunks = [
            c if i == 0 else chunks[i - 1][-overlap:] + "\n" + c
            for i, c in enumerate(chunks)
        ]
    return chunks


# ---------------------------------------------------------------- 向量库

def _get_collection():
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _doc_id(source: str, index: int, text: str) -> str:
    return hashlib.sha1(f"{source}::{index}::{text}".encode()).hexdigest()[:20]


def scan_folder(folder: Path) -> list[Path]:
    return sorted(
        p for p in Path(folder).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
    )


def ingest_files(paths: list[Path], progress_cb=None) -> dict:
    """加载文件 -> 分块 -> 向量化 -> 写入向量库。重复入库同一文件会先清掉旧块。"""
    coll = _get_collection()
    stats = {"files": 0, "chunks": 0, "skipped": []}
    for i, path in enumerate(paths):
        path = Path(path)
        try:
            text = load_file(path)
        except Exception as exc:
            stats["skipped"].append(f"{path.name}: {exc}")
            continue
        chunks = chunk_text(text)
        if not chunks:
            stats["skipped"].append(f"{path.name}: 内容为空")
            continue
        source = path.name
        coll.delete(where={"source": source})
        coll.add(
            ids=[_doc_id(source, j, c) for j, c in enumerate(chunks)],
            documents=chunks,
            embeddings=embed_texts(chunks),
            metadatas=[{"source": source, "chunk": j} for j in range(len(chunks))],
        )
        stats["files"] += 1
        stats["chunks"] += len(chunks)
        if progress_cb:
            progress_cb(i + 1, len(paths), source)
    return stats


def query(question: str, top_k: int = 4) -> list[dict]:
    coll = _get_collection()
    if coll.count() == 0:
        return []
    result = coll.query(
        query_embeddings=embed_texts([question]),
        n_results=min(top_k, coll.count()),
    )
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "chunk": meta["chunk"],
                "score": round(1 - dist, 4),  # 余弦距离 -> 相似度
            }
        )
    return hits


def list_sources() -> list[dict]:
    coll = _get_collection()
    if coll.count() == 0:
        return []
    summary: dict[str, int] = {}
    for meta in coll.get(include=["metadatas"])["metadatas"]:
        summary[meta["source"]] = summary.get(meta["source"], 0) + 1
    return [{"source": k, "chunks": v} for k, v in sorted(summary.items())]


def clear() -> None:
    coll = _get_collection()
    ids = coll.get()["ids"]
    if ids:
        coll.delete(ids=ids)


# ---------------------------------------------------------------- 生成

SYSTEM_PROMPT = (
    "你是个人知识库问答助手。严格依据给定资料回答问题:"
    "用到某条资料时在句末标注来源编号,如 [1];"
    "资料不足以回答时直接说明不知道,不要编造。"
)


def llm_configured() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def stream_answer(question: str, contexts: list[dict]):
    """流式生成回答,yield 文本片段,供 st.write_stream 消费。"""
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL") or None,
        api_key=os.getenv("LLM_API_KEY"),
    )
    material = "\n\n".join(
        f"[{i + 1}] 来源《{c['source']}》:\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    stream = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "glm-4-flash"),
        temperature=0.3,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"资料:\n{material}\n\n问题:{question}"},
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
