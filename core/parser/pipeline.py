# core/parser/pipeline.py
"""
文档解析与分块主流程。
串联：文件扫描 → PDF 解析 → 语义分块 → 输出 JSON
"""
import json
import logging
from pathlib import Path
from typing import List

from core.config import DATA_DIR, OUTPUT_DIR
from core.parser.pdf_parser import DeepDocPdfParser
from core.parser.chunker import SemanticChunker
from core.utils.file_utils import scan_pdf_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_parser: DeepDocPdfParser,
    chunker: SemanticChunker,
    file_meta: dict,
    output_dir: Path,
    force_reprocess: bool = False,
) -> List[dict]:
    """
    处理单个 PDF 文件：解析 → 分块 → 保存为 JSON。

    Args:
        pdf_parser: DeepDocPdfParser 实例
        chunker: SemanticChunker 实例
        file_meta: 文件元数据（来自 scan_pdf_files）
        output_dir: JSON 输出目录
        force_reprocess: True 则忽略缓存，重新解析

    Returns:
        chunk 字典列表（可直接送入向量化流程）
    """
    output_path = output_dir / f"{file_meta['fiscal_period']}_chunks.json"

    # 若已处理且不强制重新处理，直接读取缓存
    if output_path.exists() and not force_reprocess:
        logger.info(f"[缓存命中] {output_path.name}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"[处理中] {file_meta['source_file']} ({file_meta['fiscal_period']})")

    try:
        # Step 1: PDF 解析
        parse_result = pdf_parser.parse(file_meta["file_path"])

        # Step 2: 语义分块
        chunks = chunker.chunk(
            paragraphs=parse_result["paragraphs"],
            tables=parse_result["tables"],
            base_metadata=file_meta,
        )

        # Step 3: 序列化为 dict
        chunk_dicts = [c.to_dict() for c in chunks]

        # Step 4: 保存 JSON（供调试和缓存）
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

        logger.info(f"[完成] {file_meta['fiscal_period']}: {len(chunks)} 个 chunk → {output_path}")
        return chunk_dicts

    except Exception as e:
        logger.error(f"[失败] {file_meta['source_file']}: {e}", exc_info=True)
        return []


def run_pipeline(
    data_dir: str = str(DATA_DIR),
    output_dir: str = str(OUTPUT_DIR),
    force_reprocess: bool = False,
) -> List[dict]:
    """
    全量处理流程入口。

    Args:
        data_dir: PDF 文件目录
        output_dir: JSON 输出目录
        force_reprocess: True 则忽略缓存，重新解析所有文件

    Returns:
        所有文档的 chunk 列表（合并）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 扫描 PDF 文件
    pdf_files = scan_pdf_files(data_dir)
    if not pdf_files:
        logger.warning(f"在 {data_dir} 中未找到符合命名规范的 PDF 文件！")
        logger.warning("请将文件放入 data/ 目录，并命名为 tesla_10k_YYYY.pdf 或 tesla_10q_YYYY_qN.pdf")
        return []

    logger.info(f"发现 {len(pdf_files)} 个财报文件，开始处理...")

    pdf_parser = DeepDocPdfParser()
    chunker = SemanticChunker()

    all_chunks: List[dict] = []
    for file_meta in pdf_files:
        chunks = process_single_pdf(
            pdf_parser=pdf_parser,
            chunker=chunker,
            file_meta=file_meta,
            output_dir=output_path,
            force_reprocess=force_reprocess,
        )
        all_chunks.extend(chunks)

    # 输出汇总统计
    stats = _compute_stats(all_chunks)
    logger.info("=" * 60)
    logger.info(f"全量处理完成！总计 {len(all_chunks)} 个 chunk")
    logger.info(f"  - 文本 chunk: {stats['text_count']}")
    logger.info(f"  - 表格 chunk: {stats['table_count']}")
    logger.info(f"  - 涉及文档: {stats['documents']}")
    logger.info(f"  - 平均 chunk token 数: {stats['avg_tokens']:.0f}")
    logger.info("=" * 60)

    return all_chunks


def _compute_stats(chunks: List[dict]) -> dict:
    if not chunks:
        return {"text_count": 0, "table_count": 0, "documents": [], "avg_tokens": 0}

    text_count = sum(1 for c in chunks if c["metadata"]["chunk_type"] == "text")
    table_count = sum(1 for c in chunks if c["metadata"]["chunk_type"] == "table")
    documents = list(set(c["metadata"]["fiscal_period"] for c in chunks))
    avg_tokens = sum(c.get("token_count", 0) for c in chunks) / len(chunks)

    return {
        "text_count": text_count,
        "table_count": table_count,
        "documents": sorted(documents),
        "avg_tokens": avg_tokens,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="特斯拉财报解析与分块")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="PDF 目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="JSON 输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存，重新处理")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        force_reprocess=args.force,
    )
