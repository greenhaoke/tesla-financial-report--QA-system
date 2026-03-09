# test_embedding.py
"""
Embedding 流程测试脚本。

包含：
  1. 单元测试（不依赖模型下载）：工厂函数、metadata 转换
  2. 集成测试（需下载/已有本地模型）：端到端嵌入 + ChromaDB 写入验证

用法：
  # 仅单元测试（无需模型）
  conda run -n tesla python test_embedding.py

  # 集成测试（需先安装 sentence-transformers 并下载模型）
  conda run -n tesla python test_embedding.py --integration

  # 指定单文件集成测试
  conda run -n tesla python test_embedding.py --integration --json-file output/FY2025_chunks.json
"""

import sys
import json
import argparse
import tempfile
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# 单元测试
# ═══════════════════════════════════════════════════════════════

def test_chroma_metadata_conversion():
    """验证 _to_chroma_metadata 的转换规则（None → 默认值，list → JSON 字符串）。"""
    from core.embedder.embed_pipeline import _to_chroma_metadata

    print("\n[TEST] _to_chroma_metadata 转换规则")

    meta = {
        "chunk_id": "FY2025_p1_0",
        "quarter": None,
        "table_title": None,
        "table_index": None,
        "section_hierarchy": ["Part I", "Item 1", "Business"],
        "source_file": "tesla_10k_2025.pdf",
        "chunk_type": "text",
        "year": 2025,
    }
    converted = _to_chroma_metadata(meta)

    assert converted["quarter"] == -1, f"FAIL: quarter 应为 -1，得 {converted['quarter']}"
    assert converted["table_title"] == "", f"FAIL: table_title 应为 空串，得 {converted['table_title']}"
    assert converted["table_index"] == -1, f"FAIL: table_index 应为 -1，得 {converted['table_index']}"
    assert isinstance(converted["section_hierarchy"], str), "FAIL: section_hierarchy 应为字符串"
    parsed = json.loads(converted["section_hierarchy"])
    assert parsed == ["Part I", "Item 1", "Business"], f"FAIL: section_hierarchy 内容不符: {parsed}"
    # 不修改原始 meta
    assert isinstance(meta["section_hierarchy"], list), "FAIL: 原始 meta 不应被修改"

    print("  ✓ quarter: None → -1")
    print("  ✓ table_title: None → ''")
    print("  ✓ table_index: None → -1")
    print("  ✓ section_hierarchy: list → JSON string（原始数据未被修改）")


def test_config_constants():
    """验证 config.py 中的嵌入相关配置是否正确加载。"""
    from core.config import EMBEDDING_CONFIG, CHROMA_COLLECTION, DB_DATA_DIR

    print("\n[TEST] config.py 嵌入配置")

    assert "provider" in EMBEDDING_CONFIG, "FAIL: EMBEDDING_CONFIG 缺少 'provider'"
    assert "local_model" in EMBEDDING_CONFIG, "FAIL: EMBEDDING_CONFIG 缺少 'local_model'"
    assert "batch_size" in EMBEDDING_CONFIG, "FAIL: EMBEDDING_CONFIG 缺少 'batch_size'"
    assert isinstance(CHROMA_COLLECTION, str) and CHROMA_COLLECTION, "FAIL: CHROMA_COLLECTION 无效"
    assert DB_DATA_DIR.name == "chroma", f"FAIL: DB_DATA_DIR 应在 chroma 子目录，得 {DB_DATA_DIR}"

    print(f"  ✓ provider: {EMBEDDING_CONFIG['provider']}")
    print(f"  ✓ local_model: {EMBEDDING_CONFIG['local_model']}")
    print(f"  ✓ batch_size: {EMBEDDING_CONFIG['batch_size']}")
    print(f"  ✓ CHROMA_COLLECTION: {CHROMA_COLLECTION}")
    print(f"  ✓ DB_DATA_DIR: {DB_DATA_DIR}")


def test_factory_invalid_provider():
    """验证工厂函数对非法 provider 的拒绝。"""
    from core.embedder.factory import get_embedder

    print("\n[TEST] factory.get_embedder 非法 provider 拒绝")
    try:
        get_embedder({"provider": "unknown_provider"})
        assert False, "FAIL: 应抛出 ValueError"
    except ValueError as e:
        print(f"  ✓ 正确抛出 ValueError: {e}")


def test_base_embedder_interface():
    """验证 BaseEmbedder 无法直接实例化（抽象类）。"""
    from core.embedder.base_embedder import BaseEmbedder

    print("\n[TEST] BaseEmbedder 抽象类检查")
    try:
        _ = BaseEmbedder()
        assert False, "FAIL: BaseEmbedder 不应可直接实例化"
    except TypeError:
        print("  ✓ BaseEmbedder 无法直接实例化（正确）")


# ═══════════════════════════════════════════════════════════════
# 集成测试（需要 sentence-transformers 和模型文件）
# ═══════════════════════════════════════════════════════════════

def test_local_embedder_basic(model_name: str, device: str = None):
    """验证 LocalEmbedder 基本功能：形状、归一化、GPU 检测。返回已加载的 embedder 供复用，避免重复加载。"""
    print(f"\n[TEST] LocalEmbedder 基本功能（模型: {model_name}）")

    from core.embedder.local_embedder import LocalEmbedder, _detect_device

    # 设备检测
    detected = _detect_device(None)
    print(f"  ✓ 自动检测设备: {detected}")

    embedder = LocalEmbedder(model_name_or_path=model_name, device=device)
    print(f"  ✓ 模型加载完成，设备: {embedder.device}")

    # 单文本
    vec = embedder.embed_one("Tesla reported record revenues in 2025.")
    assert isinstance(vec, list) and len(vec) > 0, "FAIL: 向量应为非空列表"
    print(f"  ✓ embed_one() 返回向量，维度: {len(vec)}")

    # 批量
    texts = [
        "Tesla Model Y is a compact SUV.",
        "Energy storage revenue grew 67% year over year.",
        "Robotaxi service launched in Q2 2025.",
    ]
    vecs = embedder.embed(texts)
    assert len(vecs) == 3, f"FAIL: 应返回 3 个向量，得 {len(vecs)}"
    assert all(len(v) == len(vec) for v in vecs), "FAIL: 所有向量维度应一致"
    print(f"  ✓ embed([3条]) 返回 3 个维度为 {len(vec)} 的向量")

    # 维度属性
    assert embedder.embedding_dim == len(vec), "FAIL: embedding_dim 属性不一致"
    print(f"  ✓ embedding_dim: {embedder.embedding_dim}")

    # 空列表
    empty = embedder.embed([])
    assert empty == [], "FAIL: 空输入应返回空列表"
    print("  ✓ embed([]) 返回 []")

    # 返回 embedder 供后续测试复用，避免重复加载占用 VRAM
    return embedder


def test_end_to_end_single_file(json_path: str, model_name: str, device: str = None, embedder=None):
    """
    端到端测试：读取 JSON → 嵌入 → 写入临时 ChromaDB → 查询验证。

    Args:
        embedder: 可选，传入已加载的 LocalEmbedder 实例以复用，
                  避免重复加载大模型导致 OOM（对 bge-m3 等大模型至关重要）
    """
    print(f"\n[TEST] 端到端单文件嵌入（{Path(json_path).name}）")

    from core.embedder.embed_pipeline import embed_single_file
    from core.embedder.local_embedder import LocalEmbedder

    # ★ 关键：只在未传入 embedder 时才创建新实例
    if embedder is None:
        print(f"  [注意] 未传入 embedder，将创建新实例（model: {model_name}）")
        embedder = LocalEmbedder(model_name_or_path=model_name, device=device)

    embed_cfg = {
        "provider": "local",
        "local_model": model_name,
        "device": device,
        # bge-m3 在 8GB 显存上建议 batch_size=16 以防 OOM
        "batch_size": 16,
    }

    import gc
    import chromadb

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        stat = embed_single_file(
            json_path=json_path,
            db_dir=tmp_dir,
            collection_name="test_collection",
            embedder_config=embed_cfg,
            force_reembed=False,
            embedder=embedder,          # ★ 传入已加载的模型，不重复分配 VRAM
        )

        print(f"  ✓ 第一次嵌入: {stat}")
        assert stat["embedded"] > 0, "FAIL: 应至少嵌入一个 chunk"
        assert stat["skipped"] == 0, "FAIL: 第一次不应有跳过"
        assert stat["collection_count"] == stat["embedded"]

        # 增量：再次运行，应全部跳过
        stat2 = embed_single_file(
            json_path=json_path,
            db_dir=tmp_dir,
            collection_name="test_collection",
            embedder_config=embed_cfg,
            force_reembed=False,
            embedder=embedder,          # 同一实例
        )
        print(f"  ✓ 第二次嵌入（增量跳过）: {stat2}")
        assert stat2["embedded"] == 0, "FAIL: 增量模式应跳过所有已处理 chunk"
        assert stat2["skipped"] == stat["embedded"]

        # ChromaDB 查询验证（复用同一个 embedder，不重复加载模型）
        client = chromadb.PersistentClient(path=tmp_dir)
        collection = client.get_collection("test_collection")
        count = collection.count()
        assert count > 0, "FAIL: collection 应有数据"

        query_vec = embedder.embed_one("Tesla revenue and financial performance")
        results = collection.query(query_embeddings=[query_vec], n_results=3)
        assert len(results["ids"][0]) == 3, "FAIL: 查询应返回 3 条结果"
        print(f"  ✓ ChromaDB 查询成功，返回 {len(results['ids'][0])} 条结果")
        for doc in results["documents"][0]:
            print(f"    - {doc[:80]}...")

        # ★ Windows 关键：退出 with 块前显式释放 ChromaDB 文件句柄
        # ChromaDB 在 Windows 中持有 .bin 文件的排他锁，不释放则 TemporaryDirectory 清理失败
        del collection, client
        gc.collect()  # 强制触发 __del__，确保文件句柄关闭

    print("  ✓ 临时目录已清理")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_unit_tests():
    print("=" * 60)
    print("运行 Embedding 单元测试（无需模型）")
    print("=" * 60)
    test_config_constants()
    test_chroma_metadata_conversion()
    test_factory_invalid_provider()
    test_base_embedder_interface()
    print("\n" + "=" * 60)
    print("✅ 所有 Embedding 单元测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tesla RAG Embedding 测试套件")
    p.add_argument("--integration", action="store_true", help="运行集成测试（需要模型）")
    p.add_argument(
        "--json-file", default=None,
        help="集成测试使用的 JSON 文件（默认自动查找 output/ 下第一个）",
    )
    p.add_argument(
        "--model", default="D:\\OneDrive\\Desktop\\大模型应用开发学习\\model\\BAAI\\bge-m3",
        help="集成测试使用的本地嵌入模型",
    )
    p.add_argument("--device", default=None, help="设备（cuda/cpu/mps），None 表示自动检测")
    args = p.parse_args()

    # 单元测试（始终运行）
    run_unit_tests()

    if args.integration:
        print("\n" + "=" * 60)
        print("运行集成测试（需要 sentence-transformers 和模型）")
        print("=" * 60)

        # ★ 创建一次 embedder，后续所有测试共享（避免 bge-m3 等大模型 OOM）
        loaded_embedder = test_local_embedder_basic(args.model, args.device)

        # 选择 JSON 文件
        if args.json_file:
            json_file = args.json_file
        else:
            output_dir = PROJECT_ROOT / "output"
            candidates = sorted(output_dir.glob("*_chunks.json"))
            if not candidates:
                print("  [SKIP] output/ 目录下没有 *_chunks.json 文件，跳过端到端测试")
                sys.exit(0)
            json_file = str(candidates[0])
            print(f"  自动选择: {json_file}")

        # 传入已加载的 embedder，整个测试流程只有一份模型在 VRAM
        test_end_to_end_single_file(json_file, args.model, args.device, embedder=loaded_embedder)

        print("\n" + "=" * 60)
        print("✅ 所有集成测试通过！")
        print("=" * 60)
    else:
        print("\n提示：传入 --integration 可运行端到端嵌入测试（需已安装 sentence-transformers）。")
