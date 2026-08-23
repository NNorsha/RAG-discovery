"""命令行提问:python ask.py "你的问题" """
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出

import rag_core as rag


def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python ask.py "你的问题"')
        return
    question = " ".join(sys.argv[1:])
    contexts = rag.query(question, top_k=4)
    if not contexts:
        print("知识库为空,请先运行 python ingest.py")
        return
    if not rag.llm_configured():
        print("未配置 LLM,仅展示检索结果:")
        for i, c in enumerate(contexts, 1):
            print(f"[{i}] {c['source']} 片段{c['chunk']} (相似度 {c['score']})")
        return
    for delta in rag.stream_answer(question, contexts):
        print(delta, end="", flush=True)
    print("\n\n来源:")
    for i, c in enumerate(contexts, 1):
        print(f"[{i}] {c['source']} 片段{c['chunk']} (相似度 {c['score']})")


if __name__ == "__main__":
    main()
