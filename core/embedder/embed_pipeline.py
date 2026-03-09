# core/embedder/embed_pipeline.py
"""
Embedding 主流程。

功能：
  - 读取分块 JSON 文件 → 批量嵌入 content → 将向量 + metadata 写入 ChromaDB

支持两种调用方式：
  1. embed_single_file(json_path, ...)  — 单文件嵌入
  2. run_embed_pipeline(json_dir, ...)  — 扫描目录，批量处理所有 *_chunks.json

ChromaDB 持久化路径：db_data/chroma（由 DB_DATA_DIR 配置）
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

# 禁用 ChromaDB 匿名遥测，避免向 posthog.com 发送数据（消除 SSL 重试日志）
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)


# ─── 内部辅助函数 ────────────────────────────────────────────────────────────

def _to_chroma_metadata(meta: dict) -> dict:
    """
    将 chunk 的 metadata 字典转换为 ChromaDB 兼容格式。
    规则与 ChunkMetadata.to_chroma_metadata() 保持一致：
      - None 的 quarter     → -1
      - None 的 table_title → ""
      - None 的 table_index → -1
      - list 的 section_hierarchy → JSON 字符串
    """
    m = dict(meta)  # 浅拷贝，不修改原始数据
    if isinstance(m.get("section_hierarchy"), list):
        m["section_hierarchy"] = json.dumps(m["section_hierarchy"], ensure_ascii=False)
    if m.get("quarter") is None:
        m["quarter"] = -1
    if m.get("table_title") is None:
        m["table_title"] = ""
    if m.get("table_index") is None:
        m["table_index"] = -1
    return m


def _get_or_create_collection(client, collection_name: str):
    """获取或创建 ChromaDB collection（不含嵌入函数，向量由我们自己管理）。"""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )


def _load_chunks(json_path: Union[str, Path]) -> List[dict]:
    """读取 JSON 文件，返回 chunk 列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"JSON 格式错误: {json_path}，期望列表，实际为 {type(chunks)}")
    return chunks


def _get_existing_ids(collection) -> set:
    """获取 collection 中已存在的所有 chunk_id。"""
    result = collection.get(include=[])  # 只取 ids，不取向量和文档
    return set(result["ids"])


# ─── 核心处理函数 ─────────────────────────────────────────────────────────────

def embed_single_file(
    json_path: Union[str, Path],
    db_dir: Optional[Union[str, Path]] = None,
    collection_name: Optional[str] = None,
    embedder_config: Optional[dict] = None,
    batch_size: int = 64,
    force_reembed: bool = False,
    embedder=None,
) -> dict:
    """
    对单个 JSON 分块文件进行嵌入并写入 ChromaDB。

    Args:
        json_path: 分块 JSON 文件路径（如 output/FY2025_chunks.json）
        db_dir: ChromaDB 持久化目录（None 则使用 DB_DATA_DIR 配置）
        collection_name: ChromaDB collection 名称（None 则使用 CHROMA_COLLECTION 配置）
        embedder_config: 嵌入器配置字典（None 则使用 EMBEDDING_CONFIG 配置）
        batch_size: 每批嵌入的 chunk 数量
        force_reembed: True 则忽略已有向量，强制重新嵌入
        embedder: 可选，传入已实例化的 BaseEmbedder，避免重复加载模型（节省显存）
                  若为 None 则根据 embedder_config 自动创建

    Returns:
        统计字典：{"file": str, "total": int, "embedded": int, "skipped": int, "collection_count": int}
    """
    from core.config import DB_DATA_DIR, CHROMA_COLLECTION, EMBEDDING_CONFIG
    from core.embedder.factory import get_embedder
    import chromadb

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    # 配置
    db_dir = Path(db_dir) if db_dir else DB_DATA_DIR
    collection_name = collection_name or CHROMA_COLLECTION
    embedder_config = embedder_config or EMBEDDING_CONFIG
    batch_size = embedder_config.get("batch_size", batch_size)

    logger.info(f"[EmbedPipeline] 处理文件: {json_path.name}")
    logger.info(f"[EmbedPipeline] ChromaDB 路径: {db_dir}")
    logger.info(f"[EmbedPipeline] Collection: {collection_name}")

    # 初始化
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = _get_or_create_collection(client, collection_name)
    # 若未传入外部 embedder，则按配置创建（批量场景下应在外部创建后传入以复用）
    if embedder is None:
        embedder = get_embedder(embedder_config)

    # 加载 chunks
    chunks = _load_chunks(json_path)
    logger.info(f"[EmbedPipeline] 读取 {len(chunks)} 个 chunk")

    # 确定需要处理的 chunks（增量跳过）
    if force_reembed:
        to_embed = chunks
        skipped = 0
    else:
        existing_ids = _get_existing_ids(collection)
        to_embed = [c for c in chunks if c["metadata"]["chunk_id"] not in existing_ids]
        skipped = len(chunks) - len(to_embed)
        if skipped:
            logger.info(f"[EmbedPipeline] 跳过已嵌入: {skipped} 个 chunk")

    embedded = 0
    # 分批嵌入并写入
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i: i + batch_size]
        texts = [c["content"] for c in batch]

        logger.info(
            f"[EmbedPipeline] 批次 {i // batch_size + 1}"
            f"/{(len(to_embed) + batch_size - 1) // batch_size}"
            f" — {len(batch)} 条"
        )

        vectors = embedder.embed(texts)

        ids = [c["metadata"]["chunk_id"] for c in batch]
        metadatas = [_to_chroma_metadata(c["metadata"]) for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        embedded += len(batch)

    total_in_db = collection.count()
    logger.info(
        f"[EmbedPipeline] ✅ {json_path.name} 完成 | "
        f"新增: {embedded} | 跳过: {skipped} | "
        f"Collection 总量: {total_in_db}"
    )

    return {
        "file": str(json_path),
        "total": len(chunks),
        "embedded": embedded,
        "skipped": skipped,
        "collection_count": total_in_db,
    }


def run_embed_pipeline(
    json_dir: Optional[Union[str, Path]] = None,
    db_dir: Optional[Union[str, Path]] = None,
    collection_name: Optional[str] = None,
    embedder_config: Optional[dict] = None,
    batch_size: int = 64,
    force_reembed: bool = False,
) -> List[dict]:
    """
    扫描目录下所有 *_chunks.json，依次对每个文件调用 embed_single_file()。

    关键优化：只加载一次嵌入模型并在所有文件间共享，避免大模型（如 bge-m3）
    因重复加载导致的显存溢出（OOM）。

    Args:
        json_dir: JSON 文件目录（None 则使用 OUTPUT_DIR 配置）
        db_dir: ChromaDB 持久化目录
        collection_name: ChromaDB collection 名称
        embedder_config: 嵌入器配置字典
        batch_size: 每批嵌入的 chunk 数量
        force_reembed: True 则强制重新嵌入所有 chunk

    Returns:
        每个文件的统计字典列表
    """
    from core.config import OUTPUT_DIR, EMBEDDING_CONFIG
    from core.embedder.factory import get_embedder

    json_dir = Path(json_dir) if json_dir else OUTPUT_DIR
    json_files = sorted(json_dir.glob("*_chunks.json"))

    if not json_files:
        logger.warning(f"[EmbedPipeline] 在 {json_dir} 中未找到 *_chunks.json 文件")
        return []

    logger.info(f"[EmbedPipeline] 发现 {len(json_files)} 个 JSON 文件，开始批量嵌入...")

    # ★ 关键：只创建一次 embedder，所有文件共享同一个模型实例
    cfg = embedder_config or EMBEDDING_CONFIG
    shared_embedder = get_embedder(cfg)
    logger.info(f"[EmbedPipeline] 嵌入模型已加载，将在 {len(json_files)} 个文件间共享")

    results = []
    for json_path in json_files:
        try:
            stat = embed_single_file(
                json_path=json_path,
                db_dir=db_dir,
                collection_name=collection_name,
                embedder_config=cfg,
                batch_size=batch_size,
                force_reembed=force_reembed,
                embedder=shared_embedder,   # 传入共享实例，不重复加载
            )
            results.append(stat)
        except Exception as e:
            logger.error(f"[EmbedPipeline] 处理 {json_path.name} 失败: {e}", exc_info=True)
            results.append({"file": str(json_path), "error": str(e)})

    # 汇总统计
    total_embedded = sum(r.get("embedded", 0) for r in results)
    total_skipped = sum(r.get("skipped", 0) for r in results)
    logger.info("=" * 60)
    logger.info(f"[EmbedPipeline] 全部完成！文件数: {len(results)}")
    logger.info(f"  新增嵌入: {total_embedded} | 跳过: {total_skipped}")
    logger.info("=" * 60)

    return results


# ─── CLI 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 确保项目根目录在路径中
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from core.config import DB_DATA_DIR, OUTPUT_DIR, CHROMA_COLLECTION, EMBEDDING_CONFIG

    parser = argparse.ArgumentParser(
        description="Tesla 财报 Embedding 流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 批量处理 output/ 目录下所有 JSON（本地模型，GPU 自动检测）
  python -m core.embedder.embed_pipeline

  # 处理单个 JSON 文件
  python -m core.embedder.embed_pipeline --json-file output/FY2025_chunks.json

  # 使用 API 嵌入
  python -m core.embedder.embed_pipeline --provider api --api-model text-embedding-3-small

  # 强制重新嵌入（忽略缓存）
  python -m core.embedder.embed_pipeline --force
        """,
    )

    # 互斥：单文件 vs 目录
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--json-file", default=None,
        help="处理单个 JSON 文件路径（与 --json-dir 互斥）",
    )
    src_group.add_argument(
        "--json-dir", default=str(OUTPUT_DIR),
        help=f"包含 *_chunks.json 的目录（默认: {OUTPUT_DIR}）",
    )

    # 数据库
    parser.add_argument("--db-dir", default=str(DB_DATA_DIR), help="ChromaDB 持久化目录")
    parser.add_argument("--collection", default=CHROMA_COLLECTION, help="ChromaDB collection 名称")

    # 嵌入器配置
    parser.add_argument(
        "--provider", default=EMBEDDING_CONFIG.get("provider", "local"),
        choices=["local", "api"],
        help="嵌入方式：local（sentence-transformers）或 api（OpenAI 兼容）",
    )
    parser.add_argument(
        "--model", default=None,
        help="本地模型名称/路径（provider=local 时有效）",
    )
    parser.add_argument(
        "--device", default=None,
        help="设备（cuda / cpu / mps），默认自动检测（GPU 优先）",
    )
    parser.add_argument(
        "--api-model", default=EMBEDDING_CONFIG.get("api_model", "text-embedding-3-small"),
        help="API 嵌入模型名称（provider=api 时有效）",
    )
    parser.add_argument(
        "--api-base-url", default=EMBEDDING_CONFIG.get("api_base_url", "https://api.openai.com/v1"),
        help="OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--api-key-env", default=EMBEDDING_CONFIG.get("api_key_env", "OPENAI_API_KEY"),
        help="API Key 环境变量名",
    )
    parser.add_argument(
        "--batch-size", type=int, default=EMBEDDING_CONFIG.get("batch_size", 64),
        help="每批嵌入的 chunk 数量",
    )
    parser.add_argument("--force", action="store_true", help="强制重新嵌入（忽略缓存）")

    args = parser.parse_args()

    # 构建嵌入配置
    embed_cfg = dict(EMBEDDING_CONFIG)
    embed_cfg["provider"] = args.provider
    if args.model:
        embed_cfg["local_model"] = args.model
    if args.device:
        embed_cfg["device"] = args.device
    embed_cfg["api_model"] = args.api_model
    embed_cfg["api_base_url"] = args.api_base_url
    embed_cfg["api_key_env"] = args.api_key_env
    embed_cfg["batch_size"] = args.batch_size

    # 执行
    if args.json_file:
        stat = embed_single_file(
            json_path=args.json_file,
            db_dir=args.db_dir,
            collection_name=args.collection,
            embedder_config=embed_cfg,
            batch_size=args.batch_size,
            force_reembed=args.force,
        )
        print(f"\n✅ 完成: {stat}")
    else:
        results = run_embed_pipeline(
            json_dir=args.json_dir,
            db_dir=args.db_dir,
            collection_name=args.collection,
            embedder_config=embed_cfg,
            batch_size=args.batch_size,
            force_reembed=args.force,
        )
        print(f"\n✅ 批量完成，共处理 {len(results)} 个文件")
