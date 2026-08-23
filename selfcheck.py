"""冒烟测试:对已入库内容跑几个检索,确认召回正常。

用法: python selfcheck.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出

import rag_core as rag

QUESTIONS = [
    "RAG 和微调应该怎么选?",
    "HNSW 是什么?",
    "怎么控制模型幻觉?",
]


def main() -> None:
    sources = rag.list_sources()
    print(f"知识库:{len(sources)} 个文档\n")
    for q in QUESTIONS:
        print(f"Q: {q}")
        hits = rag.query(q, top_k=2)
        if not hits:
            print("  (无结果,知识库为空?)")
        for i, h in enumerate(hits, 1):
            text = h["text"][:60].replace("\n", " ")
            print(f"  [{i}] {h['source']} (相似度 {h['score']}) {text}…")
        print()


if __name__ == "__main__":
    main()
