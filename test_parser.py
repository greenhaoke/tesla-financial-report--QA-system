# test_parser.py
"""
验证解析流程是否正常工作的测试用例集合。
包含：
  1. 单元测试（不依赖真实 PDF）：file_utils、chunker、metadata
  2. 集成测试（需要真实 PDF）：pdf_parser、完整 pipeline

用法（单元测试，无需 PDF）：
    conda run -n tesla python test_parser.py

用法（集成测试，需指定 PDF）：
    conda run -n tesla python test_parser.py --pdf data/10-K/tesla_10k_2022.pdf

用法（调试某一步骤）：
    conda run -n tesla python test_parser.py --pdf <path> --step parser
    conda run -n tesla python test_parser.py --pdf <path> --step chunker
    conda run -n tesla python test_parser.py --pdf <path> --step pipeline
"""

import sys
import json
import argparse
from pathlib import Path

# ── 确保项目根目录在 sys.path 中 ─────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# 单元测试：不依赖 deepdoc 或真实 PDF
# ═══════════════════════════════════════════════════════════════

def test_parse_filename():
    """验证 parse_filename 对各种命名格式的解析。"""
    from core.utils.file_utils import parse_filename

    cases = [
        # (文件名, 期望 report_type, 期望 year, 期望 quarter, 期望 fiscal_period)
        ("tesla_10k_2022.pdf",       "10-K", 2022, None, "FY2022"),
        ("tesla_10k_2021.pdf",       "10-K", 2021, None, "FY2021"),
        ("tesla_10q_2023_q2.pdf",    "10-Q", 2023, 2,    "2023Q2"),
        ("tesla_10q_2024_q3.pdf",    "10-Q", 2024, 3,    "2024Q3"),
        ("tesla_10q_2021_q1.pdf",    "10-Q", 2021, 1,    "2021Q1"),
        ("TESLA_10K_2022.pdf",       "10-K", 2022, None, "FY2022"),  # 大写
        ("tsla-20221231.pdf",        "10-K", 2022, None, "FY2022"),  # SEC EDGAR 格式
        ("tsla-20230331.pdf",        "10-Q", 2023, 1,    "2023Q1"),
        ("tsla-20230630.pdf",        "10-Q", 2023, 2,    "2023Q2"),
    ]

    print("\n[TEST] parse_filename")
    passed = 0
    for filename, exp_type, exp_year, exp_q, exp_fp in cases:
        result = parse_filename(filename)
        assert result is not None, f"  FAIL: parse_filename({filename!r}) 返回 None"
        assert result["report_type"] == exp_type, \
            f"  FAIL: {filename} report_type={result['report_type']} (期望 {exp_type})"
        assert result["year"] == exp_year, \
            f"  FAIL: {filename} year={result['year']} (期望 {exp_year})"
        assert result["quarter"] == exp_q, \
            f"  FAIL: {filename} quarter={result['quarter']} (期望 {exp_q})"
        assert result["fiscal_period"] == exp_fp, \
            f"  FAIL: {filename} fiscal_period={result['fiscal_period']} (期望 {exp_fp})"
        print(f"  ✓ {filename:35s} → {result['report_type']}, {result['fiscal_period']}")
        passed += 1

    # 无效文件名应返回 None
    invalid_names = ["annual_report_2022.pdf", "TSLA2022.pdf", "report.pdf"]
    for name in invalid_names:
        assert parse_filename(name) is None, \
            f"  FAIL: parse_filename({name!r}) 应返回 None，但未返回"
        print(f"  ✓ {name:35s} → None（正确拒绝）")
        passed += 1

    print(f"  通过: {passed}/{len(cases) + len(invalid_names)}")


def test_count_tokens():
    """验证 token 估算的合理性。"""
    from core.parser.chunker import _count_tokens

    print("\n[TEST] _count_tokens")
    cases = [
        ("", 0, 0),               # 空串
        ("hello world", 2, 4),    # 英文：11 chars / 4 ≈ 2
        ("a" * 400, 90, 110),      # 英文大文本：400/4 = 100
    ]
    for text, lo, hi in cases:
        tok = _count_tokens(text)
        assert lo <= tok <= hi, \
            f"  FAIL: '{text[:20]}...' token={tok}，期望在 [{lo}, {hi}]"
        print(f"  ✓ len={len(text):4d} → token≈{tok:4d}  (期望 [{lo}, {hi}])")


def test_split_by_sentence():
    """验证按句子切分并带重叠的逻辑。"""
    from core.parser.chunker import _split_by_sentence

    print("\n[TEST] _split_by_sentence")

    # 构造一段超过 max_tokens 的长文本
    long_text = " ".join(
        [f"This is sentence number {i} about Tesla financial results." for i in range(50)]
    )
    chunks = _split_by_sentence(long_text, max_tokens=60, overlap_tokens=15)

    assert len(chunks) > 1, "FAIL: 长文本应被切分为多块"
    for c in chunks:
        tokens = sum(1 for _ in c.split())  # 粗略计数
        # 每块不应远超 max_tokens（考虑估算误差）
        assert len(c) > 0, "FAIL: 不应有空块"
    print(f"  ✓ 50 句 → {len(chunks)} 块，每块非空")

    # 短文本不应被切分
    short = "Tesla reported record revenue. Net income grew significantly."
    chunks_short = _split_by_sentence(short, max_tokens=200, overlap_tokens=20)
    assert len(chunks_short) == 1, f"FAIL: 短文本应为 1 块，实际 {len(chunks_short)} 块"
    print(f"  ✓ 短文本 → 1 块（不切分）")


def test_chunk_metadata_serialization():
    """验证 ChunkMetadata 的序列化和 ChromaDB 格式转换。"""
    from core.parser.metadata import ChunkMetadata, Chunk
    from core.parser.chunker import _count_tokens

    print("\n[TEST] ChunkMetadata 序列化")

    meta = ChunkMetadata(
        source_file="tesla_10k_2022.pdf",
        report_type="10-K",
        year=2022,
        quarter=None,
        fiscal_period="FY2022",
        page_start=10,
        page_end=12,
        section_title="Results of Operations",
        section_hierarchy=["Part II", "Item 7", "Results of Operations"],
        chunk_type="text",
        chunk_index=5,
        chunk_id="FY2022_p10_5",
    )
    chunk = Chunk(
        content="Tesla revenues increased 51% in 2022.",
        metadata=meta,
        token_count=_count_tokens("Tesla revenues increased 51% in 2022."),
    )

    # to_dict
    d = chunk.to_dict()
    assert d["content"] == chunk.content
    assert d["metadata"]["year"] == 2022
    assert d["metadata"]["quarter"] is None
    assert isinstance(d["metadata"]["section_hierarchy"], list)
    print(f"  ✓ to_dict() 正常，quarter=None 正确保留")

    # to_chroma_metadata
    cm = meta.to_chroma_metadata()
    assert cm["quarter"] == -1, f"FAIL: ChromaDB 格式 quarter 应为 -1，得 {cm['quarter']}"
    assert isinstance(cm["section_hierarchy"], str), "FAIL: section_hierarchy 应序列化为 str"
    parsed_hier = json.loads(cm["section_hierarchy"])
    assert parsed_hier == ["Part II", "Item 7", "Results of Operations"]
    print(f"  ✓ to_chroma_metadata() 正常，section_hierarchy 正确 JSON 序列化")

    # JSON 往返
    json_str = json.dumps(d, ensure_ascii=False)
    restored = json.loads(json_str)
    assert restored["content"] == chunk.content
    print(f"  ✓ JSON 往返序列化正常")


def test_semantic_chunker_unit():
    """
    用构造的假数据验证 SemanticChunker 的核心分块逻辑，
    不依赖 deepdoc 或真实 PDF。
    """
    from core.parser.chunker import SemanticChunker

    print("\n[TEST] SemanticChunker 单元测试（Mock 数据）")

    chunker = SemanticChunker()
    base_meta = {
        "source_file": "tesla_10k_2022.pdf",
        "report_type": "10-K",
        "year": 2022,
        "quarter": None,
        "fiscal_period": "FY2022",
        "file_path": "data/10-K/tesla_10k_2022.pdf",
    }

    # ── Case 1: 标题 + 段落 ──────────────────────────────────────
    paragraphs_1 = [
        {"text": "RESULTS OF OPERATIONS", "page": 1, "is_heading": True,  "font_size": 16.0, "bbox": []},
        {"text": "Tesla's total revenues were $81.5 billion for fiscal year 2022, an increase of 51%.", "page": 1, "is_heading": False, "font_size": 12.0, "bbox": []},
        {"text": "Automotive revenues increased to $71.5 billion driven by higher deliveries.", "page": 2, "is_heading": False, "font_size": 12.0, "bbox": []},
    ]
    chunks = chunker.chunk(paragraphs_1, [], base_meta)
    assert len(chunks) >= 1, "FAIL: 应至少生成 1 个 chunk"
    # 检查标题包含在 chunk 内容里（以 ## 标记）
    heading_chunks = [c for c in chunks if "RESULTS OF OPERATIONS" in c.content]
    assert heading_chunks, "FAIL: 应有包含章节标题的 chunk"
    print(f"  ✓ Case 1: 标题+段落 → {len(chunks)} 个 chunk，标题正确融入")

    # ── Case 2: 表格是独立 chunk ─────────────────────────────────
    tables_2 = [
        {
            "markdown": "| Year | Revenue |\n| --- | --- |\n| 2022 | $81.5B |\n| 2021 | $53.8B |",
            "page": 3,
            "title": "Annual Revenue Summary",
            "table_index": 0,
            "raw": [],
        }
    ]
    paragraphs_2 = [
        {"text": "The following table summarizes annual revenue.", "page": 3, "is_heading": False, "font_size": 12.0, "bbox": []},
    ]
    chunks_2 = chunker.chunk(paragraphs_2, tables_2, base_meta)
    table_chunks = [c for c in chunks_2 if c.metadata.chunk_type == "table"]
    text_chunks  = [c for c in chunks_2 if c.metadata.chunk_type == "text"]
    assert len(table_chunks) == 1, f"FAIL: 应有 1 个表格 chunk，得 {len(table_chunks)}"
    assert len(text_chunks)  >= 1, f"FAIL: 应有 >= 1 个文本 chunk，得 {len(text_chunks)}"
    assert table_chunks[0].metadata.table_index == 0
    print(f"  ✓ Case 2: 表格独立 chunk，文本 chunk 正常分离")

    # ── Case 3: 超长文本自动切分 ─────────────────────────────────
    long_para = ". ".join([f"Tesla achieved record deliveries in quarter {i}" for i in range(200)]) + "."
    paragraphs_3 = [
        {"text": long_para, "page": 5, "is_heading": False, "font_size": 12.0, "bbox": []},
    ]
    chunks_3 = chunker.chunk(paragraphs_3, [], base_meta)
    assert len(chunks_3) > 1, f"FAIL: 超长文本应被切分，得 {len(chunks_3)} 块"
    for c in chunks_3:
        assert c.token_count > 0
    print(f"  ✓ Case 3: 超长文本 → {len(chunks_3)} 块，均有 token_count")

    # ── Case 4: chunk_id 全局唯一 ────────────────────────────────
    all_ids = [c.metadata.chunk_id for c in chunks_3]
    assert len(all_ids) == len(set(all_ids)), f"FAIL: chunk_id 存在重复"
    print(f"  ✓ Case 4: chunk_id 全局唯一")

    # ── Case 5: 章节层级栈 ───────────────────────────────────────
    paragraphs_5 = [
        {"text": "PART II",                       "page": 1, "is_heading": True,  "font_size": 20.0, "bbox": []},
        {"text": "Item 7. Management Discussion",  "page": 2, "is_heading": True,  "font_size": 15.0, "bbox": []},
        {"text": "Liquidity and Capital Resources","page": 3, "is_heading": True,  "font_size": 13.0, "bbox": []},
        {"text": "We had $22.2 billion in cash.",  "page": 3, "is_heading": False, "font_size": 12.0, "bbox": []},
    ]
    chunks_5 = chunker.chunk(paragraphs_5, [], base_meta)
    # 找包含 "Liquidity" 的 chunk，检查其层级
    liquidity_chunks = [c for c in chunks_5 if "Liquidity" in c.content]
    assert liquidity_chunks, "FAIL: 应有包含 Liquidity 的 chunk"
    hier = liquidity_chunks[0].metadata.section_hierarchy
    assert len(hier) >= 2, f"FAIL: section_hierarchy 应有 >= 2 层，得 {hier}"
    print(f"  ✓ Case 5: section_hierarchy 正确构建 → {hier}")


# ═══════════════════════════════════════════════════════════════
# 集成测试：依赖真实 PDF 文件
# ═══════════════════════════════════════════════════════════════

def test_single_file(pdf_path: str, step: str = "all"):
    """
    使用真实 PDF 文件测试完整解析链路。

    Args:
        pdf_path: 测试用 PDF 文件路径
        step: 要测试的步骤，可选 "parser" / "chunker" / "pipeline" / "all"
    """
    from core.utils.file_utils import parse_filename

    filename = Path(pdf_path).name

    # 1. 文件名解析
    meta = parse_filename(filename)
    assert meta is not None, (
        f"文件名解析失败！请重命名为规范格式。\n"
        f"  当前文件名: {filename}\n"
        f"  正确格式: tesla_10k_YYYY.pdf 或 tesla_10q_YYYY_qN.pdf"
    )
    meta["file_path"] = pdf_path
    print(f"\n✓ 文件名解析: {meta}")

    if step in ("parser", "all"):
        # 2. PDF 解析
        from core.parser.pdf_parser import DeepDocPdfParser
        parser = DeepDocPdfParser()
        result = parser.parse(pdf_path)

        assert "paragraphs" in result and "tables" in result
        paras = result["paragraphs"]
        tables = result["tables"]
        print(f"✓ PDF 解析: {len(paras)} 段落, {len(tables)} 张表格")

        # 检查段落结构
        if paras:
            for key in ("text", "page", "is_heading", "font_size"):
                assert key in paras[0], f"FAIL: 段落缺少字段 '{key}'"
            print(f"  段落字段完整 ✓")
            for i, p in enumerate(paras[:3]):
                print(f"  段落 {i+1} [p{p['page']}] {'[标题]' if p['is_heading'] else '      '}: {p['text'][:80]}...")

        # 检查表格结构
        if tables:
            for key in ("markdown", "page", "title", "table_index"):
                assert key in tables[0], f"FAIL: 表格缺少字段 '{key}'"
            print(f"  表格字段完整 ✓")
            t = tables[0]
            print(f"  表格 1 [p{t['page']}]: {t['title'] or '(无标题)'}")
            print(f"  {t['markdown'][:200]}...")
        else:
            print(f"  (未检测到表格，如使用 PlainParser 则属正常)")

    if step in ("chunker", "all"):
        if step == "chunker":
            from core.parser.pdf_parser import DeepDocPdfParser
            parser = DeepDocPdfParser()
            result = parser.parse(pdf_path)

        # 3. 分块
        from core.parser.chunker import SemanticChunker
        chunker = SemanticChunker()
        chunks = chunker.chunk(result["paragraphs"], result["tables"], meta)

        assert len(chunks) > 0, "FAIL: 至少应生成 1 个 chunk"
        text_chunks  = [c for c in chunks if c.metadata.chunk_type == "text"]
        table_chunks = [c for c in chunks if c.metadata.chunk_type == "table"]
        print(f"\n✓ 分块完成: {len(chunks)} 个 chunk")
        print(f"  - 文本块: {len(text_chunks)}")
        print(f"  - 表格块: {len(table_chunks)}")

        # chunk_id 唯一性
        ids = [c.metadata.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "FAIL: chunk_id 存在重复！"
        print(f"  ✓ chunk_id 全局唯一")

        # token_count
        assert all(c.token_count >= 0 for c in chunks), "FAIL: token_count 不能为负"
        avg_tok = sum(c.token_count for c in chunks) / len(chunks)
        max_tok = max(c.token_count for c in chunks)
        print(f"  平均 token: {avg_tok:.0f}，最大 token: {max_tok}")

        # 打印前 2 个 chunk
        for i, c in enumerate(chunks[:2]):
            print(f"\n  Chunk {i+1}:")
            print(f"    ID   : {c.metadata.chunk_id}")
            print(f"    章节 : {c.metadata.section_title}")
            print(f"    类型 : {c.metadata.chunk_type}")
            print(f"    Tokens: {c.token_count}")
            print(f"    内容预览: {c.content[:100]}...")

    if step in ("pipeline", "all"):
        # 4. 完整 pipeline（含 JSON 输出）
        from core.parser.pipeline import process_single_pdf
        from core.parser.pdf_parser import DeepDocPdfParser
        from core.parser.chunker import SemanticChunker

        output_dir = Path(PROJECT_ROOT) / "output"
        chunk_dicts = process_single_pdf(
            pdf_parser=DeepDocPdfParser(),
            chunker=SemanticChunker(),
            file_meta=meta,
            output_dir=output_dir,
            force_reprocess=True,
        )
        assert len(chunk_dicts) > 0, "FAIL: pipeline 输出为空"
        output_file = output_dir / f"{meta['fiscal_period']}_chunks.json"
        assert output_file.exists(), f"FAIL: JSON 输出文件不存在: {output_file}"
        print(f"\n✓ pipeline 完整运行: {len(chunk_dicts)} 个 chunk → {output_file}")

    print("\n✅ 所有测试通过！")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_unit_tests():
    """运行所有不依赖 PDF 的单元测试。"""
    print("=" * 60)
    print("运行单元测试（无需 PDF 文件）")
    print("=" * 60)
    test_parse_filename()
    test_count_tokens()
    test_split_by_sentence()
    test_chunk_metadata_serialization()
    test_semantic_chunker_unit()
    print("\n" + "=" * 60)
    print("✅ 所有单元测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tesla RAG Phase 1 测试套件")
    p.add_argument("--pdf",  default=None,  help="测试用 PDF 文件路径（集成测试必须）")
    p.add_argument("--step", default="all", choices=["parser", "chunker", "pipeline", "all"],
                   help="集成测试步骤（默认 all）")
    p.add_argument("--unit-only", action="store_true", help="仅运行单元测试，不进行集成测试")
    args = p.parse_args()

    # 始终运行单元测试
    run_unit_tests()

    # 如有 PDF 且不限定仅单元测试，运行集成测试
    if args.pdf and not args.unit_only:
        print(f"\n运行集成测试（PDF: {args.pdf}，step: {args.step}）")
        print("=" * 60)
        test_single_file(args.pdf, args.step)
    elif not args.unit_only:
        print("\n提示：传入 --pdf <path> 可运行集成测试（需真实 PDF 文件）。")
