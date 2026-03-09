# core/parser/chunker.py
import re
import logging
from typing import List, Optional
from core.parser.metadata import Chunk, ChunkMetadata
from core.config import CHUNK_CONFIG

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    """快速估算 token 数（英文约 4 chars/token，中文约 1.5 chars/token）"""
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    non_chinese = len(text) - chinese
    return int(chinese / 1.5 + non_chinese / 4)


def _split_by_sentence(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    当单个段落超过 max_tokens 时，按句子边界切分并添加重叠。

    为确保覆盖常见的英文句子结束符（.!?），使用 lookbehind 断点。
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # 保留末尾若干句作为重叠
            overlap_buffer: List[str] = []
            buf_tokens = 0
            for s in reversed(current_sentences):
                s_tok = _count_tokens(s)
                if buf_tokens + s_tok <= overlap_tokens:
                    overlap_buffer.insert(0, s)
                    buf_tokens += s_tok
                else:
                    break
            current_sentences = overlap_buffer + [sent]
            current_tokens = sum(_count_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sent)
            current_tokens += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


class SemanticChunker:
    """
    基于章节感知的语义分块器。

    输入：deepdoc 规范化后的 paragraphs + tables
    输出：Chunk 列表（含完整元数据）

    分块策略（Section-Aware Semantic Chunking）：
    1. 识别章节标题 → 每个新标题开启新块的积累窗口
    2. 同一章节内，将连续短段落合并，直到 token 数接近上限（512）
    3. 超过上限时，按句子边界切断，保留 64 token 的滑动重叠
    4. 每张表格无论大小，始终作为独立 chunk，不与文本合并
    5. 每个 chunk 记录 section_hierarchy（章节路径），供过滤使用
    """

    def __init__(self):
        self.max_tokens = CHUNK_CONFIG["max_tokens_per_chunk"]
        self.min_tokens = CHUNK_CONFIG["min_tokens_to_merge"]
        self.overlap_tokens = CHUNK_CONFIG["overlap_tokens"]

    def chunk(
        self,
        paragraphs: List[dict],
        tables: List[dict],
        base_metadata: dict,
    ) -> List[Chunk]:
        """
        主入口：将段落+表格转换为 Chunk 列表。

        Args:
            paragraphs: pdf_parser 返回的段落列表
            tables: pdf_parser 返回的表格列表
            base_metadata: 来自文件名解析，含 report_type/year/quarter/fiscal_period/source_file

        Returns:
            List[Chunk]（按页码顺序排列）
        """
        # 合并所有元素，按页码排序
        # 表格排在同页文本之后（先处理文本以建立章节上下文）
        elements = []
        for p in paragraphs:
            elements.append({"type": "paragraph", "data": p, "page": p["page"]})
        for t in tables:
            elements.append({"type": "table", "data": t, "page": t["page"]})
        elements.sort(key=lambda x: (x["page"], 1 if x["type"] == "table" else 0))

        chunks: List[Chunk] = []
        chunk_index = 0

        # 章节层级栈 & 当前章节标题
        section_stack: List[str] = []
        current_section = "Document Header"

        # 文本积累 buffer
        buffer_texts: List[str] = []
        buffer_tokens: int = 0
        buffer_page_start: int = 1
        buffer_page_end: int = 1

        def flush_buffer():
            nonlocal buffer_texts, buffer_tokens, buffer_page_start, buffer_page_end, chunk_index

            if not buffer_texts:
                return

            full_text = "\n\n".join(buffer_texts)

            # 如果 buffer 太长，按句子再切分
            if buffer_tokens > self.max_tokens:
                sub_chunks = _split_by_sentence(full_text, self.max_tokens, self.overlap_tokens)
            else:
                sub_chunks = [full_text]

            for sub in sub_chunks:
                sub = sub.strip()
                if not sub:
                    continue
                meta = ChunkMetadata(
                    source_file=base_metadata["source_file"],
                    report_type=base_metadata["report_type"],
                    year=base_metadata["year"],
                    quarter=base_metadata.get("quarter"),
                    fiscal_period=base_metadata["fiscal_period"],
                    page_start=buffer_page_start,
                    page_end=buffer_page_end,
                    section_title=current_section,
                    section_hierarchy=list(section_stack),
                    chunk_type="text",
                    chunk_index=chunk_index,
                    chunk_id=f"{base_metadata['fiscal_period']}_p{buffer_page_start}_{chunk_index}",
                )
                chunks.append(Chunk(
                    content=sub,
                    metadata=meta,
                    token_count=_count_tokens(sub),
                ))
                chunk_index += 1

            buffer_texts = []
            buffer_tokens = 0
            buffer_page_start = buffer_page_end

        for element in elements:
            page = element["page"]

            if element["type"] == "paragraph":
                para = element["data"]
                text = para["text"].strip()
                if not text:
                    continue

                # 识别到新标题：先 flush 当前 buffer，然后更新章节
                if para["is_heading"]:
                    flush_buffer()
                    buffer_page_start = page
                    self._update_section_stack(section_stack, text, para["font_size"])
                    current_section = text
                    # 标题本身加入下一个 chunk 的开头（以 ## 标记）
                    buffer_texts.append(f"## {text}")
                    buffer_tokens += _count_tokens(text)
                    buffer_page_end = page
                    continue

                tok = _count_tokens(text)

                # buffer 加上这段会超限 → 先 flush（但不足 min_tokens 的 buffer 继续积累）
                if buffer_tokens + tok > self.max_tokens and buffer_tokens >= self.min_tokens:
                    flush_buffer()
                    buffer_page_start = page

                buffer_texts.append(text)
                buffer_tokens += tok
                buffer_page_end = page

            elif element["type"] == "table":
                # 表格：先 flush 文本 buffer，再单独输出表格 chunk
                flush_buffer()
                buffer_page_start = page

                table = element["data"]
                markdown = table["markdown"]
                title = table.get("title", "") or ""

                # 在 markdown 前加上表格标题
                content = f"**{title}**\n\n{markdown}" if title else markdown

                # 使用 pdf_parser 提供的精确结束页（page_end），兼容旧格式
                table_page_end = table.get("page_end", page)

                meta = ChunkMetadata(
                    source_file=base_metadata["source_file"],
                    report_type=base_metadata["report_type"],
                    year=base_metadata["year"],
                    quarter=base_metadata.get("quarter"),
                    fiscal_period=base_metadata["fiscal_period"],
                    page_start=page,
                    page_end=table_page_end,
                    section_title=current_section,
                    section_hierarchy=list(section_stack),
                    chunk_type="table",
                    table_title=title if title else None,
                    table_index=table["table_index"],
                    chunk_index=chunk_index,
                    chunk_id=f"{base_metadata['fiscal_period']}_p{page}_t{table['table_index']}",
                )
                chunks.append(Chunk(
                    content=content,
                    metadata=meta,
                    token_count=_count_tokens(content),
                ))
                chunk_index += 1

        # 处理最后剩余的 buffer
        flush_buffer()

        logger.info(f"分块完成: 共 {len(chunks)} 个 chunk（文本+表格）")
        return chunks

    def _update_section_stack(self, stack: List[str], heading: str, font_size: float):
        """
        根据字体大小维护章节层级栈（字体越大 → 层级越高）。

        层级规则（字号区间为估算）：
          >= 18  → 顶层（Part I、Part II 等）：清空栈
          >= 14  → 二级（Item 7、Item 8 等）：弹出最后一级
          <  14  → 三级或更深（子节）：追加，最多保留 3 层
        """
        if font_size >= 18:
            stack.clear()
            stack.append(heading)
        elif font_size >= 14:
            # 二级标题：如果栈深度 > 1，弹出顶层再压入
            if len(stack) > 1:
                stack.pop()
            stack.append(heading)
        else:
            # 三级标题：追加，超过 3 层则替换顶层
            if len(stack) < 3:
                stack.append(heading)
            else:
                stack[-1] = heading
