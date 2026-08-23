"""一次性开发工具:把 BGE-small-zh 从 PyTorch 导出为 ONNX 并做精度校验。

正常使用不需要本脚本 —— fastembed 会自动从 HuggingFace 下载 ONNX 模型。
仅当网络无法访问 HuggingFace 时(如国内环境):
  1. 从 ModelScope 镜像下载模型文件到 models/bge-small-zh/
     https://modelscope.cn/models/BAAI/bge-small-zh-v1.5/files
     需要: config.json / tokenizer.json / tokenizer_config.json /
           special_tokens_map.json / vocab.txt / model.safetensors
  2. pip install torch transformers
  3. python export_onnx.py
"""
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast

MODEL_DIR = Path(__file__).resolve().parent / "models" / "bge-small-zh"
ONNX_PATH = MODEL_DIR / "model_optimized.onnx"  # fastembed 约定的文件名
INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]


def main() -> None:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertModel.from_pretrained(MODEL_DIR).eval()

    dummy = tokenizer("测试输入", return_tensors="pt")
    torch.onnx.export(
        model,
        tuple(dummy[k] for k in INPUT_NAMES),
        ONNX_PATH,
        input_names=INPUT_NAMES,
        output_names=["last_hidden_state"],
        dynamic_axes={
            **{k: {0: "batch", 1: "seq"} for k in INPUT_NAMES},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    print(f"导出完成: {ONNX_PATH.name} ({ONNX_PATH.stat().st_size / 1e6:.1f}MB)")

    # 校验 1:ONNX 与 PyTorch 的归一化 CLS 向量余弦相似度应约等于 1
    import onnxruntime as ort

    text = "检索增强生成可以解决模型幻觉问题"
    batch = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        ref = model(**batch).last_hidden_state[:, 0]
    ref = torch.nn.functional.normalize(ref, p=2, dim=1).numpy()

    sess = ort.InferenceSession(str(ONNX_PATH))
    out = sess.run(None, {k: v.numpy() for k, v in batch.items()})[0][:, 0]
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    sim = float((ref @ out.T)[0, 0])
    print(f"ONNX vs PyTorch 余弦相似度: {sim:.6f}")
    assert sim > 0.9999, "导出精度异常"

    # 校验 2:fastembed 全链路加载结果一致
    from fastembed import TextEmbedding

    emb = next(
        iter(
            TextEmbedding(
                "BAAI/bge-small-zh-v1.5", specific_model_path=str(MODEL_DIR)
            ).embed([text])
        )
    )
    sim2 = float(np.dot(emb, ref[0]))
    print(f"fastembed vs PyTorch 余弦相似度: {sim2:.6f}")
    assert sim2 > 0.9999, "fastembed 加载结果异常"
    print("全部通过。model.safetensors 仅导出时需要,此后可删除。")


if __name__ == "__main__":
    main()
