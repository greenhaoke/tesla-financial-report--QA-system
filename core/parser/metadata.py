# core/parser/metadata.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import json


@dataclass
class ChunkMetadata:
    # ── 来源信息 ──────────────────────────────────────────
    source_file: str          # 原始文件名，如 "tesla_10k_2022.pdf"
    report_type: str          # "10-K" 或 "10-Q"
    year: int                 # 财年，如 2022
    quarter: Optional[int]    # 季报季次：1/2/3/4，年报为 None
    fiscal_period: str        # 可读标签，如 "FY2022" 或 "2023Q2"

    # ── 位置信息 ──────────────────────────────────────────
    page_start: int           # 起始页（1-indexed）
    page_end: int             # 结束页（1-indexed）
    section_title: str        # 所属章节标题，如 "Liquidity and Capital Resources"
    section_hierarchy: list   # 章节层级路径，如 ["Item 7", "Liquidity", "Cash Flow"]

    # ── 内容类型 ──────────────────────────────────────────
    chunk_type: Literal["text", "table", "mixed"]
    table_title: Optional[str] = None   # 表格标题（仅 chunk_type="table" 时有值）
    table_index: Optional[int] = None   # 表格在文档中的序号

    # ── 块标识 ────────────────────────────────────────────
    chunk_id: str = ""        # 格式："{fiscal_period}_{page}_{index}"
    chunk_index: int = 0      # 在文档内的全局序号

    def to_dict(self) -> dict:
        return asdict(self)

    def to_chroma_metadata(self) -> dict:
        """转换为 ChromaDB 兼容格式（值只能是 str/int/float/bool）"""
        d = self.to_dict()
        d["section_hierarchy"] = json.dumps(d["section_hierarchy"])
        if d["quarter"] is None:
            d["quarter"] = -1  # ChromaDB 不支持 None
        if d["table_title"] is None:
            d["table_title"] = ""
        if d["table_index"] is None:
            d["table_index"] = -1
        return d


@dataclass
class Chunk:
    content: str              # 文本内容（表格转为 Markdown 格式）
    metadata: ChunkMetadata
    token_count: int = 0      # 估算 token 数（用于分块控制）

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "token_count": self.token_count,
        }
