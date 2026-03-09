# core/parser/pdf_parser.py
"""
调用 deepdoc 解析 PDF 财报，提取结构化的文本段落和表格。

deepdoc 有两种解析器：
  - RAGFlowPdfParser: 完整解析器，使用 OCR + 布局识别 + 表格检测（需要 ML 模型）
    返回: (text_string_with_position_tags, [(img, table_data), ...])
  - PlainParser: 轻量级解析器，基于 pypdf 文本提取
    返回: ([(text, ""), ...], [])

本模块优先尝试 RAGFlowPdfParser，如模型加载失败则回退到 PlainParser。
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── 财报常见章节标题关键词（用于标题识别） ─────────────────────
FINANCIAL_SECTION_KEYWORDS = [
    # 10-K / 10-Q 标准章节
    r"^item\s+\d+[a-z]?\.",
    r"^part\s+[ivxlcdm]+",
    # 财务分析常见小节
    r"liquidity",
    r"capital resources",
    r"results of operations",
    r"gross margin",
    r"automotive",
    r"revenue",
    r"cash flow",
    r"risk factor",
    r"management.{0,10}discussion",
    r"critical accounting",
    r"segment",
    r"outlook",
    r"forward.looking",
    r"consolidated balance",
    r"consolidated statements",
    r"notes to",
    r"quantitative and qualitative",
]

HEADING_PATTERN = re.compile(
    "|".join(FINANCIAL_SECTION_KEYWORDS),
    re.IGNORECASE
)

# 匹配 deepdoc 位置标签：@@页码\tx0\tx1\ttop\tbottom##
_POSITION_TAG_RE = re.compile(r"@@([\d\-]+)\t([0-9.]+)\t([0-9.]+)\t([0-9.]+)\t([0-9.]+)##")


def _remove_position_tags(text: str) -> str:
    """去除 deepdoc 内嵌的位置标签，返回纯文本。"""
    return _POSITION_TAG_RE.sub("", text).strip()


def _extract_page_from_tag(text: str, default: int = 1) -> int:
    """从 deepdoc 位置标签中提取第一个出现的页码。"""
    m = _POSITION_TAG_RE.search(text)
    if m:
        # 标签格式：@@页码(-页码)\t...##，取第一个页码
        pages = m.group(1).split("-")
        try:
            return int(pages[0])
        except (ValueError, IndexError):
            pass
    return default


class DeepDocPdfParser:
    """
    封装 deepdoc 的 PDF 解析能力。

    职责：
    - 调用 deepdoc 接口获取原始解析结果
    - 将原始结果规范化为统一的内部格式
    - 识别章节标题，构建层级
    - 将表格转换为 Markdown
    """

    def __init__(self):
        self._parser = None
        self._parser_type = None
        self._load_deepdoc_parser()

    def _load_deepdoc_parser(self):
        """
        动态加载 deepdoc 解析器。
        优先使用 RAGFlowPdfParser（完整功能），失败则回退到 PlainParser（轻量）。
        """
        # 方式 A：完整解析器（需要 OCR / ML 模型）
        try:
            from core.deepdoc.parser.pdf_parser import RAGFlowPdfParser
            self._parser = RAGFlowPdfParser()
            self._parser_type = "ragflow"
            logger.info("deepdoc RAGFlowPdfParser 加载成功（完整模式）")
            return
        except Exception as e:
            logger.warning(f"RAGFlowPdfParser 加载失败（{e}），尝试 PlainParser...")

        # 方式 B：轻量级解析器（仅文字提取，无 OCR）
        try:
            from core.deepdoc.parser.pdf_parser import PlainParser
            self._parser = PlainParser()
            self._parser_type = "plain"
            logger.info("deepdoc PlainParser 加载成功（轻量模式，无表格识别）")
            return
        except Exception as e:
            raise ImportError(
                f"无法加载任何 deepdoc 解析器，请检查 core/deepdoc 目录。错误：{e}"
            )

    # ──────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        解析一个 PDF 文件，返回规范化的解析结果。

        Returns:
            {
                "paragraphs": [
                    {
                        "text": str,          # 段落文字
                        "page": int,          # 页码（1-indexed）
                        "is_heading": bool,   # 是否是章节标题
                        "font_size": float,   # 字体大小估算
                        "bbox": list,         # 坐标（若有）
                    },
                    ...
                ],
                "tables": [
                    {
                        "markdown": str,      # Markdown 格式的表格
                        "page": int,          # 所在页（起始页，1-indexed）
                        "page_end": int,      # 所在页（结束页，1-indexed）
                        "title": str,         # 表格标题（若能识别）
                        "table_index": int,   # 在文档中的序号（0-indexed）
                        "raw": list,          # 原始行列数据
                    },
                    ...
                ]
            }
        """
        logger.info(f"开始解析: {pdf_path}（使用 {self._parser_type} 模式）")

        if self._parser_type == "ragflow":
            paragraphs, tables = self._parse_ragflow(pdf_path)
        else:
            raw_result = self._parser(pdf_path)
            paragraphs = self._normalize_plain_paragraphs(raw_result)
            tables = []  # PlainParser 不提取表格

        logger.info(f"解析完成: {len(paragraphs)} 段落, {len(tables)} 张表格")
        return {"paragraphs": paragraphs, "tables": tables}

    def _parse_ragflow(self, pdf_path: str):
        """
        对 RAGFlowPdfParser 分步调用，以 need_position=True 获取表格精确页码。
        返回 (paragraphs, tables)。
        """
        parser = self._parser
        # 步骤 1：加载图像 / OCR
        parser.__images__(pdf_path, 3)
        # 步骤 2：版面识别
        parser._layouts_rec(3)
        # 步骤 3：表格结构识别
        parser._table_transformer_job(3)
        # 步骤 4：文本合并
        parser._text_merge()
        # 步骤 5：段落拼接
        parser._concat_downward()
        # 步骤 6：过滤目录页
        parser._filter_forpages()
        # 步骤 7：提取表格/图形，传入 need_position=True 以获取页码
        from copy import deepcopy
        try:
            tbls_with_pos = parser._extract_table_figure(
                need_image=True, ZM=3,
                return_html=False, need_position=True
            )
            # tbls_with_pos: [((img, table_data), [(page_number, x0, x1, top, bottom), ...]), ...]
            text_str = parser._RAGFlowPdfParser__filterout_scraps(deepcopy(parser.boxes), 3)
            raw_result = (text_str, None)  # tbls 单独处理
            paragraphs = self._normalize_ragflow_paragraphs(raw_result)
            tables = self._normalize_ragflow_tables_with_pos(tbls_with_pos)
        except Exception as e:
            logger.warning(f"带位置信息的表格提取失败（{e}），回退到普通模式")
            # 回退：重新完整解析
            raw_result = self._parser(pdf_path)
            paragraphs = self._normalize_ragflow_paragraphs(raw_result)
            tables = self._normalize_ragflow_tables(raw_result)
        return paragraphs, tables

    # ──────────────────────────────────────────────────────────────────
    # RAGFlowPdfParser 适配
    # RAGFlowPdfParser.__call__ 返回:
    #   (text_str, tbls)
    #   text_str: 多段落拼接的字符串，每段用 \n\n 分隔，含 @@页码\t...## 位置标签
    #   tbls: [(PIL.Image, table_data), ...] 其中 table_data 是 list[list[str]]
    # ──────────────────────────────────────────────────────────────────

    def _normalize_ragflow_paragraphs(self, raw_result) -> List[Dict]:
        """从 RAGFlowPdfParser 返回结果中提取段落。"""
        if not isinstance(raw_result, (tuple, list)) or len(raw_result) < 1:
            return []

        text_str = raw_result[0]
        if not isinstance(text_str, str):
            return []

        paragraphs = []
        # 按双换行分段
        raw_paras = text_str.split("\n\n")

        for para_text in raw_paras:
            para_text = para_text.strip()
            if not para_text:
                continue

            page = _extract_page_from_tag(para_text)
            clean_text = _remove_position_tags(para_text)

            if not clean_text:
                continue

            # 粗略估算字体大小（RAGFlow 不直接提供字号，用启发规则）
            font_size = self._estimate_font_size(clean_text)

            paragraphs.append({
                "text": clean_text,
                "page": page,
                "is_heading": self._is_heading(clean_text, font_size),
                "font_size": font_size,
                "bbox": [],
            })

        return paragraphs

    def _normalize_ragflow_tables(self, raw_result) -> List[Dict]:
        """
        从 RAGFlowPdfParser 返回结果中提取表格（不带位置信息的回退路径）。
        raw_result[1] = [(PIL.Image, table_data), ...]
        table_data 通常为 list[list[str]]（已由 TableStructureRecognizer.construct_table 处理）
        """
        if not isinstance(raw_result, (tuple, list)) or len(raw_result) < 2:
            return []

        raw_tables = raw_result[1]
        if not raw_tables:
            return []

        tables = []
        for idx, item in enumerate(raw_tables):
            # item 格式：(PIL.Image, table_data) 或直接 table_data
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                table_data = item[1]
            else:
                table_data = item

            markdown = self._table_to_markdown(table_data)
            if not markdown:
                continue

            tables.append({
                "markdown": markdown,
                "page": 1,       # 无位置信息时的默认值
                "page_end": 1,
                "title": "",
                "table_index": idx,
                "raw": table_data if isinstance(table_data, list) else [],
            })

        return tables

    def _normalize_ragflow_tables_with_pos(
        self, tbls_with_pos
    ) -> List[Dict]:
        """
        从带位置信息的表格列表中提取表格及其精确页码。

        tbls_with_pos 格式（need_position=True 时）：
            [((img, table_data), [(page_number, x0, x1, top, bottom), ...]), ...]

        positions 中的 page_number 是 0-indexed（deepdoc 内部用 page_from 偏移后的值）。
        deepdoc 源码: poss.append((pn + self.page_from, left, right, top, bott))
        其中 pn 本身已是 0-indexed page index，所以最终值也是 0-indexed。
        转换为 1-indexed：page_number + 1。
        """
        if not tbls_with_pos:
            return []

        tables = []
        tbl_idx = 0  # 全局表格计数（跳过图形）
        for entry in tbls_with_pos:
            # entry = ((img, table_data), positions_list)
            if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
                continue
            res_item, positions = entry

            # res_item = (img, table_data) 或仅 table_data
            if isinstance(res_item, (tuple, list)) and len(res_item) >= 2:
                table_data = res_item[1]
            else:
                table_data = res_item

            # 仅处理真正的表格（table_data 是 list，图形 table_data 是字符串列表）
            # deepdoc 把 figure 和 table 都放进 res，按照: figure 在前，table 在后
            # 但 table_data 类型相同（list[list[str]]），无法区分图形文字与表格
            # 直接按顺序处理所有 entry
            markdown = self._table_to_markdown(table_data)
            if not markdown:
                tbl_idx += 1
                continue

            # 从 positions 提取页码（0-indexed → 1-indexed）
            page_start = 1
            page_end = 1
            if positions:
                try:
                    # positions: [(page_number_0indexed, x0, x1, top, bottom), ...]
                    page_numbers = [int(p[0]) + 1 for p in positions if p]
                    if page_numbers:
                        page_start = min(page_numbers)
                        page_end = max(page_numbers)
                except Exception:
                    pass

            tables.append({
                "markdown": markdown,
                "page": page_start,
                "page_end": page_end,
                "title": "",
                "table_index": tbl_idx,
                "raw": table_data if isinstance(table_data, list) else [],
            })
            tbl_idx += 1

        return tables

    # ──────────────────────────────────────────────────────────────────
    # PlainParser 适配
    # PlainParser.__call__ 返回:
    #   ([(text, ""), ...], [])
    # ──────────────────────────────────────────────────────────────────

    def _normalize_plain_paragraphs(self, raw_result) -> List[Dict]:
        """从 PlainParser 返回结果中提取段落（逐行）。"""
        if not isinstance(raw_result, (tuple, list)) or len(raw_result) < 1:
            return []

        lines_list = raw_result[0]
        if not isinstance(lines_list, list):
            return []

        paragraphs = []
        # PlainParser 按页返回 [(text_line, ""), ...]，没有页码信息
        # 我们用行计数粗略估算页码（假设约 50 行/页）
        LINES_PER_PAGE = 50

        buffer: List[str] = []
        current_page = 1

        for i, item in enumerate(lines_list):
            # 更新当前页估算
            current_page = max(1, i // LINES_PER_PAGE + 1)

            if isinstance(item, (tuple, list)):
                text = str(item[0]).strip() if item else ""
            elif isinstance(item, str):
                text = item.strip()
            else:
                continue

            if not text:
                # 空行：flush 当前 buffer
                if buffer:
                    full_text = " ".join(buffer)
                    font_size = self._estimate_font_size(full_text)
                    paragraphs.append({
                        "text": full_text,
                        "page": current_page,
                        "is_heading": self._is_heading(full_text, font_size),
                        "font_size": font_size,
                        "bbox": [],
                    })
                    buffer = []
                continue

            buffer.append(text)

            # 如果行末有句子结束符，flush
            if text.endswith((".", "!", "?", ";")) and len(buffer) >= 3:
                full_text = " ".join(buffer)
                font_size = self._estimate_font_size(full_text)
                paragraphs.append({
                    "text": full_text,
                    "page": current_page,
                    "is_heading": self._is_heading(full_text, font_size),
                    "font_size": font_size,
                    "bbox": [],
                })
                buffer = []

        # flush 剩余
        if buffer:
            full_text = " ".join(buffer)
            font_size = self._estimate_font_size(full_text)
            paragraphs.append({
                "text": full_text,
                "page": current_page,
                "is_heading": self._is_heading(full_text, font_size),
                "font_size": font_size,
                "bbox": [],
            })

        return paragraphs

    # ──────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────

    def _table_to_markdown(self, table_data) -> str:
        """将表格数据转换为 Markdown 格式。"""
        if isinstance(table_data, str):
            # construct_table 在 return_html=True 时返回 HTML，否则应为二维列表
            # 若已经是字符串直接返回
            return table_data.strip()

        rows: List[Any] = []
        if isinstance(table_data, dict):
            rows = table_data.get("rows", table_data.get("data", table_data.get("cells", [])))
        elif isinstance(table_data, list):
            rows = table_data
        else:
            return ""

        if not rows:
            return ""

        md_rows: List[str] = []
        for i, row in enumerate(rows):
            if isinstance(row, list):
                cells = [str(cell).strip().replace("|", "\\|") for cell in row]
            elif isinstance(row, dict):
                cells = [str(v).strip() for v in row.values()]
            elif isinstance(row, str):
                cells = [row.strip()]
            else:
                cells = [str(row)]

            if not cells:
                continue
            md_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:  # 表头分隔线
                md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(md_rows)

    def _is_heading(self, text: str, font_size: float = 12.0) -> bool:
        """判断一段文字是否是章节标题。"""
        text = text.strip()
        if not text:
            return False

        from core.config import CHUNK_CONFIG
        max_chars = CHUNK_CONFIG.get("heading_max_chars", 120)

        # 字体较大，且长度合理
        if font_size >= 14 and len(text) <= max_chars:
            return True
        # 匹配财报章节关键词
        if HEADING_PATTERN.search(text) and len(text) <= max_chars:
            return True
        # 全大写短文本（如 "CONSOLIDATED BALANCE SHEETS"）
        if text.isupper() and 3 <= len(text) <= 80:
            return True
        return False

    def _estimate_font_size(self, text: str) -> float:
        """
        在 PlainParser 模式下无字体信息，用启发规则估算。
        短、全大写 → 估算为标题字号；否则正文字号。
        """
        stripped = text.strip()
        if not stripped:
            return 12.0
        # 全大写且较短 → 大标题
        if stripped.isupper() and len(stripped) <= 80:
            return 16.0
        # 匹配财报章节关键词且较短 → 小标题
        if HEADING_PATTERN.search(stripped) and len(stripped) <= 120:
            return 14.0
        return 12.0
