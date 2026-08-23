"""命令行入库:python ingest.py [文件夹或文件 ...](默认扫描 ./data)"""
import sys
from pathlib import Path

import rag_core as rag


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]] or [Path(__file__).resolve().parent / "data"]
    files: list[Path] = []
    for a in args:
        files.extend(rag.scan_folder(a) if a.is_dir() else [a])
    if not files:
        print("没有找到可入库的文档(md / txt / pdf / docx)")
        return
    print(f"发现 {len(files)} 个文件,开始入库(首次运行会下载向量模型)...")
    stats = rag.ingest_files(files)
    print(f"完成:{stats['files']} 个文件,{stats['chunks']} 个分块")
    for s in stats["skipped"]:
        print(f"  跳过 {s}")


if __name__ == "__main__":
    main()
