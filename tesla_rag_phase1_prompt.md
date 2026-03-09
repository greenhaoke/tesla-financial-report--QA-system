# Claude Code 执行提示词
## 特斯拉财报智能问答系统 · 第一阶段：文档解析与智能分块

---

## ⚠️ 环境规范（最高优先级，所有命令必须遵守）

### 使用的 Python 环境：`tesla`（conda 虚拟环境）

**所有终端命令（bash/shell）必须在 `tesla` 环境下执行。**

在任何需要运行 Python、pip install、pytest 的地方，都必须按以下方式执行：

```bash
# ✅ 正确方式：每条命令都通过 conda run 在 tesla 环境中执行
conda run -n tesla python test_parser.py --pdf data/10-K/tesla_10k_2022.pdf
conda run -n tesla pip install <包名>
conda run -n tesla python -m core.parser.pipeline

# ❌ 错误方式：不能直接调用系统 python 或 pip
python test_parser.py
pip install <包名>
```

> 原因：项目使用独立的 `tesla` conda 环境，避免污染系统环境，也确保依赖版本可控。

---

### 依赖安装原则

**每次发现缺少依赖时，执行以下两步，缺一不可：**

1. **安装到 tesla 环境**
```bash
conda run -n tesla pip install <包名>
# 或者使用 conda 安装（优先 pip，conda 作为备选）
conda install -n tesla -c conda-forge <包名>
```

2. **同步写入 `requirements.txt`**（追加，不覆盖已有内容）
```
# 在 requirements.txt 末尾追加，带版本号：
<包名>>=<当前安装的版本>
```

**如何查询刚装好的版本：**
```bash
conda run -n tesla pip show <包名> | grep Version
```

---

### 初始化环境检查（Step 0 必做）

在做任何其他事情之前，先执行以下命令确认环境状态：

```bash
# 1. 确认 tesla 环境存在
conda env list | grep tesla

# 2. 查看当前已安装的包
conda run -n tesla pip list

# 3. 确认 Python 版本
conda run -n tesla python --version

# 4. 查看 deepdoc 所需依赖（如有 requirements 文件）
find core/deepdoc -name "requirements*.txt" | xargs cat 2>/dev/null

# 5. 如果 deepdoc 有 requirements，批量安装到 tesla 环境
# conda run -n tesla pip install -r core/deepdoc/requirements.txt
```

---

### requirements.txt 维护规范

项目根目录的 `requirements.txt` 是**唯一的依赖记录文件**，格式如下：

```
# ── deepdoc / ragflow 核心依赖 ────────────────────────────
# （从 deepdoc 自带的 requirements 合并过来）

# ── 解析与分块模块依赖 ────────────────────────────────────
# （本阶段新增的包写在这里）

# ── 向量数据库依赖 ────────────────────────────────────────
# （后续阶段添加）

# ── 前端/API 依赖 ─────────────────────────────────────────
# （后续阶段添加）
```

每次安装新包后，立即按分类追加到对应区块，并备注用途，例如：
```
tiktoken>=0.5.2        # token 数量估算，用于分块大小控制
chromadb>=0.4.0        # 向量数据库
```

---

## 项目背景与目标

构建一个能够处理特斯拉 2021–2025 年所有 10-K 年报和 10-Q 季报的智能问答系统。
本阶段目标：**将 PDF 财报解析为结构化数据块，为后续向量化和检索做好准备。**

---

## 工程目录规范（必须严格遵守）

```
项目根目录/
├── app/                        # 前端代码（本阶段暂不涉及）
├── core/                       # 后端业务代码
│   ├── deepdoc/                # 已有！ragflow 的解析器，只调用，不修改
│   ├── parser/                 # 【本阶段新建】解析与分块模块
│   │   ├── __init__.py
│   │   ├── pdf_parser.py       # 调用 deepdoc，提取文本+表格
│   │   ├── chunker.py          # 语义分块策略实现
│   │   ├── metadata.py         # 元数据结构定义
│   │   └── pipeline.py         # 串联解析→分块→输出的主流程
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py       # 文件扫描、路径处理工具
│   └── config.py               # 全局配置（路径、分块参数等）
├── db_data/                    # ChromaDB 向量数据库存储目录
│   └── chroma/                 # chroma 持久化文件存放位置
├── data/                       # 原始 PDF 财报文件放置目录
│   ├── 10-K/                   # 年报
│   │   ├── tesla_10k_2021.pdf
│   │   ├── tesla_10k_2022.pdf
│   │   └── ...
│   └── 10-Q/                   # 季报
│       ├── tesla_10q_2021_q1.pdf
│       └── ...
├── output/                     # 解析结果的 JSON 中间文件（调试用）
├── requirements.txt
└── README.md
```

---

## 第一步：理解 deepdoc 接口（必须先做）

在写任何代码前，**先执行以下命令**了解 deepdoc 的实际结构：

```bash
# 查看 deepdoc 目录结构
find core/deepdoc -type f -name "*.py" | head -40

# 查看主要入口文件
cat core/deepdoc/parser/__init__.py 2>/dev/null || \
cat core/deepdoc/__init__.py 2>/dev/null

# 查看 PDF 解析器
find core/deepdoc -name "*.py" | xargs grep -l "pdf\|PDF" | head -10
```

根据实际找到的接口，调用方式可能是以下之一（以实际为准）：
```python
# 可能的调用方式 A（ragflow 常见模式）
from core.deepdoc.parser import PdfParser
parser = PdfParser()
sections, tables = parser(filepath, from_page=0, to_page=None)

# 可能的调用方式 B
from core.deepdoc.parser.pdf_parser import RAGFlowPdfParser
parser = RAGFlowPdfParser()
result = parser.__call__(filepath)
```

**重要：根据 deepdoc 实际代码，适配调用接口，不要假设接口形式。**

---

## 第二步：实现 `core/config.py`

```python
# core/config.py
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
DB_DATA_DIR = ROOT_DIR / "db_data" / "chroma"

# ── 分块参数 ──────────────────────────────────────────────
CHUNK_CONFIG = {
    # 语义分块：段落合并的最大 token 数（超过则另起一块）
    "max_tokens_per_chunk": 512,
    # 段落合并的最小 token 数（太短的段落会向下合并）
    "min_tokens_to_merge": 50,
    # 块与块之间的重叠 token 数（保证上下文连续性）
    "overlap_tokens": 64,
    # 表格始终作为独立块，不与文本合并
    "table_as_standalone": True,
    # 章节标题识别的最大字符数（超过则不视为标题）
    "heading_max_chars": 120,
}

# ── 财报文档元数据 ─────────────────────────────────────────
# 文件名命名规范：tesla_{report_type}_{year}[_q{quarter}].pdf
# 例：tesla_10k_2022.pdf / tesla_10q_2023_q2.pdf
```

---

## 第三步：实现 `core/parser/metadata.py`

定义统一的元数据结构，**每一个 Chunk 都必须携带以下字段**：

```python
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
```

---

## 第四步：实现 `core/parser/pdf_parser.py`

**核心要求：**
1. 调用 deepdoc 提取文本段落和表格
2. 保留每个元素的页码信息
3. 表格转换为 Markdown 格式
4. 识别章节标题，构建层级结构

```python
# core/parser/pdf_parser.py
"""
调用 deepdoc 解析 PDF 财报，提取结构化的文本段落和表格。

deepdoc 返回的原始数据结构（以实际为准，需根据 deepdoc 代码适配）：
- 文本段落通常包含：文字内容、位置坐标、页码、字体大小等
- 表格通常包含：单元格内容、行列结构、页码
"""
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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
]

HEADING_PATTERN = re.compile(
    "|".join(FINANCIAL_SECTION_KEYWORDS),
    re.IGNORECASE
)


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
        self._load_deepdoc_parser()

    def _load_deepdoc_parser(self):
        """动态加载 deepdoc 解析器，根据实际接口适配"""
        # TODO: 根据 deepdoc 实际代码填写正确的导入路径
        # 执行前请先运行：find core/deepdoc -name "*.py" | xargs grep -l "class.*Parser"
        try:
            from core.deepdoc.parser import PdfParser
            self.parser = PdfParser()
            logger.info("deepdoc PdfParser 加载成功（方式A）")
        except ImportError:
            try:
                from core.deepdoc.parser.pdf_parser import RAGFlowPdfParser
                self.parser = RAGFlowPdfParser()
                logger.info("deepdoc RAGFlowPdfParser 加载成功（方式B）")
            except ImportError as e:
                raise ImportError(
                    f"无法加载 deepdoc 解析器，请检查 core/deepdoc 目录结构。错误：{e}"
                )

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
                        "font_size": float,   # 字体大小（若 deepdoc 提供）
                        "bbox": list,         # 坐标 [x0, y0, x1, y1]（若有）
                    },
                    ...
                ],
                "tables": [
                    {
                        "markdown": str,      # Markdown 格式的表格
                        "page": int,          # 所在页
                        "title": str,         # 表格标题（若能识别）
                        "table_index": int,   # 在文档中的序号（0-indexed）
                        "raw": list,          # 原始行列数据
                    },
                    ...
                ]
            }
        """
        logger.info(f"开始解析: {pdf_path}")
        
        # ── 调用 deepdoc ──────────────────────────────────────
        # 注意：根据 deepdoc 实际接口调整此处调用
        raw_result = self.parser(pdf_path)
        
        # ── 规范化处理 ────────────────────────────────────────
        paragraphs = self._normalize_paragraphs(raw_result)
        tables = self._normalize_tables(raw_result)
        
        logger.info(f"解析完成: {len(paragraphs)} 段落, {len(tables)} 张表格")
        return {"paragraphs": paragraphs, "tables": tables}

    def _normalize_paragraphs(self, raw_result) -> List[Dict]:
        """
        将 deepdoc 原始段落数据规范化。
        根据 deepdoc 实际输出结构适配此方法。
        """
        paragraphs = []
        # TODO: 根据 deepdoc 实际返回格式适配
        # deepdoc ragflow 通常返回 (sections, tables) 元组
        # sections 通常是包含 (text, position) 信息的列表
        
        if isinstance(raw_result, tuple):
            raw_sections = raw_result[0]  # 第一个元素通常是文本段落
        elif isinstance(raw_result, dict):
            raw_sections = raw_result.get("sections", raw_result.get("paragraphs", []))
        else:
            raw_sections = raw_result
        
        for item in raw_sections:
            if isinstance(item, str):
                text = item
                page = 0
                font_size = 12.0
            elif isinstance(item, dict):
                text = item.get("text", item.get("content", ""))
                page = item.get("page", item.get("page_number", 0))
                font_size = item.get("font_size", item.get("size", 12.0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text = item[0] if isinstance(item[0], str) else str(item[0])
                page = item[1] if isinstance(item[1], int) else 0
                font_size = item[2] if len(item) > 2 else 12.0
            else:
                continue
            
            if not text or not text.strip():
                continue
                
            paragraphs.append({
                "text": text.strip(),
                "page": max(1, int(page)),
                "is_heading": self._is_heading(text, font_size),
                "font_size": float(font_size),
                "bbox": item.get("bbox", []) if isinstance(item, dict) else [],
            })
        
        return paragraphs

    def _normalize_tables(self, raw_result) -> List[Dict]:
        """将 deepdoc 原始表格数据规范化，并转为 Markdown。"""
        tables = []
        
        if isinstance(raw_result, tuple) and len(raw_result) > 1:
            raw_tables = raw_result[1]
        elif isinstance(raw_result, dict):
            raw_tables = raw_result.get("tables", [])
        else:
            return tables
        
        for idx, table in enumerate(raw_tables):
            markdown = self._table_to_markdown(table)
            if not markdown:
                continue
            
            page = self._extract_table_page(table)
            title = self._extract_table_title(table)
            
            tables.append({
                "markdown": markdown,
                "page": page,
                "title": title or f"Table {idx + 1}",
                "table_index": idx,
                "raw": table if isinstance(table, list) else [],
            })
        
        return tables

    def _table_to_markdown(self, table) -> str:
        """将表格数据转换为 Markdown 格式。"""
        if isinstance(table, str):
            return table  # deepdoc 已经返回字符串
        
        rows = []
        if isinstance(table, dict):
            rows = table.get("rows", table.get("data", table.get("cells", [])))
        elif isinstance(table, list):
            rows = table
        
        if not rows:
            return ""
        
        md_rows = []
        for i, row in enumerate(rows):
            if isinstance(row, list):
                cells = [str(cell).strip().replace("|", "\\|") for cell in row]
            elif isinstance(row, dict):
                cells = [str(v).strip() for v in row.values()]
            else:
                cells = [str(row)]
            
            md_rows.append("| " + " | ".join(cells) + " |")
            
            if i == 0:  # 表头分隔线
                md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        
        return "\n".join(md_rows)

    def _is_heading(self, text: str, font_size: float = 12.0) -> bool:
        """判断一段文字是否是章节标题。"""
        text = text.strip()
        if not text:
            return False
        # 字体较大，且长度合理
        if font_size >= 14 and len(text) <= 120:
            return True
        # 匹配财报章节关键词
        if HEADING_PATTERN.search(text) and len(text) <= 120:
            return True
        # 全大写短文本
        if text.isupper() and 3 <= len(text) <= 80:
            return True
        return False

    def _extract_table_page(self, table) -> int:
        if isinstance(table, dict):
            return table.get("page", table.get("page_number", 1))
        return 1

    def _extract_table_title(self, table) -> Optional[str]:
        if isinstance(table, dict):
            return table.get("title", table.get("caption", None))
        return None
```

---

## 第五步：实现 `core/parser/chunker.py`

### 分块策略说明（必须理解后再实现）

**选用策略：基于章节感知的语义分块（Section-Aware Semantic Chunking）**

理由：
- 财报问题通常以章节为单位（如"MD&A 中关于毛利率的讨论"），按章节分块能显著提升召回精度
- 表格包含关键数字，必须作为独立完整单元，不能被截断
- 纯按长度分块会破坏"流动性分析"、"汽车收入拆分"等跨段落的完整论述

**具体规则：**
1. 识别章节标题 → 每个新标题开启新块的积累窗口
2. 在同一章节内，将连续短段落合并，直到 token 数接近上限（512）
3. 超过上限时，按句子边界切断，保留 64 token 的滑动重叠
4. 每张表格无论大小，始终作为独立 chunk，不与文本合并
5. 每个 chunk 记录 `section_hierarchy`（章节路径），供过滤使用

```python
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
    """当单个段落超过 max_tokens 时，按句子边界切分并添加重叠。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_sentences = []
    current_tokens = 0
    overlap_buffer = []

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # 保留末尾若干句作为重叠
            overlap_buffer = []
            buf_tokens = 0
            for s in reversed(current_sentences):
                if buf_tokens + _count_tokens(s) <= overlap_tokens:
                    overlap_buffer.insert(0, s)
                    buf_tokens += _count_tokens(s)
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
    """

    def __init__(self):
        self.max_tokens = CHUNK_CONFIG["max_tokens_per_chunk"]
        self.min_tokens = CHUNK_CONFIG["min_tokens_to_merge"]
        self.overlap_tokens = CHUNK_CONFIG["overlap_tokens"]

    def chunk(
        self,
        paragraphs: List[dict],
        tables: List[dict],
        base_metadata: dict,  # 来自文件名解析：report_type/year/quarter/fiscal_period/source_file
    ) -> List[Chunk]:
        """
        主入口：将段落+表格转换为 Chunk 列表。
        
        算法流程：
        1. 将段落和表格按页码合并排序
        2. 遍历元素，维护当前章节层级栈
        3. 段落：累积到 buffer，满了就输出
        4. 表格：立即作为独立 chunk 输出
        """
        # 合并所有元素，按页码排序
        elements = []
        for p in paragraphs:
            elements.append({"type": "paragraph", "data": p, "page": p["page"]})
        for t in tables:
            elements.append({"type": "table", "data": t, "page": t["page"]})
        elements.sort(key=lambda x: (x["page"], x["type"] == "table"))  # 表格排后（先处理文本建立章节）
        
        chunks: List[Chunk] = []
        chunk_index = 0
        
        # 章节层级栈
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
                if not sub.strip():
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
                    # 标题本身也加入下一个 chunk 的开头
                    buffer_texts.append(f"## {text}")
                    buffer_tokens += _count_tokens(text)
                    buffer_page_end = page
                    continue
                
                tok = _count_tokens(text)
                
                # buffer 加上这段会超限 → 先 flush
                if buffer_tokens + tok > self.max_tokens and buffer_tokens > self.min_tokens:
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
                title = table.get("title", "")
                
                # 在 markdown 前加上表格标题
                content = f"**{title}**\n\n{markdown}" if title else markdown
                
                meta = ChunkMetadata(
                    source_file=base_metadata["source_file"],
                    report_type=base_metadata["report_type"],
                    year=base_metadata["year"],
                    quarter=base_metadata.get("quarter"),
                    fiscal_period=base_metadata["fiscal_period"],
                    page_start=page,
                    page_end=page,
                    section_title=current_section,
                    section_hierarchy=list(section_stack),
                    chunk_type="table",
                    table_title=title,
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
        """根据字体大小维护章节层级栈（简化版：大字体=高层级）。"""
        if font_size >= 18:
            stack.clear()
            stack.append(heading)
        elif font_size >= 14:
            if len(stack) > 1:
                stack.pop()
            if not stack:
                stack.append(heading)
            else:
                stack.append(heading)
        else:
            if len(stack) < 3:
                stack.append(heading)
            else:
                stack[-1] = heading
```

---

## 第六步：实现 `core/utils/file_utils.py`

```python
# core/utils/file_utils.py
import re
from pathlib import Path
from typing import Optional, List, Dict


def parse_filename(filename: str) -> Optional[Dict]:
    """
    从文件名解析财报元数据。
    
    命名规范：tesla_{report_type}_{year}[_q{quarter}].pdf
    示例：
      tesla_10k_2022.pdf         → 10-K 年报，2022 年
      tesla_10q_2023_q2.pdf      → 10-Q 季报，2023 年 Q2
      tesla_10q_2024_q3.pdf      → 10-Q 季报，2024 年 Q3
    
    也兼容 SEC EDGAR 常见命名（如 tsla-20231231.htm 转换而来的 PDF）
    """
    name = Path(filename).stem.lower()
    
    # 标准命名
    pattern = r"tesla[_-](10[kq])[_-](\d{4})(?:[_-]q(\d))?(?:[_-].*)?"
    m = re.search(pattern, name)
    if m:
        report_type = "10-K" if "10k" in m.group(1) else "10-Q"
        year = int(m.group(2))
        quarter = int(m.group(3)) if m.group(3) else None
        fiscal_period = f"FY{year}" if report_type == "10-K" else f"{year}Q{quarter}"
        return {
            "source_file": filename,
            "report_type": report_type,
            "year": year,
            "quarter": quarter,
            "fiscal_period": fiscal_period,
        }
    
    # 兼容 tsla-YYYYMMDD 格式
    pattern2 = r"tsla[_-](\d{4})(\d{2})(\d{2})"
    m2 = re.search(pattern2, name)
    if m2:
        year = int(m2.group(1))
        month = int(m2.group(2))
        # 10-K 通常在 12 月结尾，10-Q 在 3/6/9 月
        if month == 12:
            report_type, quarter = "10-K", None
            fiscal_period = f"FY{year}"
        else:
            report_type = "10-Q"
            quarter = {3: 1, 6: 2, 9: 3}.get(month, 1)
            fiscal_period = f"{year}Q{quarter}"
        return {
            "source_file": filename,
            "report_type": report_type,
            "year": year,
            "quarter": quarter,
            "fiscal_period": fiscal_period,
        }
    
    return None


def scan_pdf_files(data_dir: str) -> List[Dict]:
    """
    递归扫描 data_dir 下的所有 PDF 文件，返回带解析元数据的列表。
    """
    results = []
    data_path = Path(data_dir)
    
    for pdf_file in sorted(data_path.rglob("*.pdf")):
        meta = parse_filename(pdf_file.name)
        if meta is None:
            print(f"[警告] 无法解析文件名，跳过: {pdf_file.name}")
            print(f"       请将文件重命名为: tesla_10k_YYYY.pdf 或 tesla_10q_YYYY_qN.pdf")
            continue
        meta["file_path"] = str(pdf_file)
        results.append(meta)
    
    return results
```

---

## 第七步：实现 `core/parser/pipeline.py`（主流程）

```python
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
    
    all_chunks = []
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
```

---

## 第八步：编写测试脚本 `test_parser.py`

在项目根目录创建，用于验证解析链路是否正常：

```python
# test_parser.py
"""
快速验证解析流程是否正常工作。
用法：python test_parser.py --pdf data/10-K/tesla_10k_2022.pdf
"""
import argparse
import json
from pathlib import Path

def test_single_file(pdf_path: str):
    from core.parser.pdf_parser import DeepDocPdfParser
    from core.parser.chunker import SemanticChunker
    from core.utils.file_utils import parse_filename
    
    # 1. 测试文件名解析
    filename = Path(pdf_path).name
    meta = parse_filename(filename)
    assert meta is not None, f"文件名解析失败！请重命名为规范格式：{filename}"
    print(f"✓ 文件名解析: {meta}")
    
    # 2. 测试 deepdoc 解析（只处理前 5 页，节省时间）
    parser = DeepDocPdfParser()
    
    # 注意：如果 deepdoc 不支持 from_page/to_page，去掉相关参数
    result = parser.parse(pdf_path)
    assert "paragraphs" in result and "tables" in result
    print(f"✓ PDF 解析: {len(result['paragraphs'])} 段落, {len(result['tables'])} 表格")
    
    # 打印前 3 个段落（验证内容）
    for i, p in enumerate(result["paragraphs"][:3]):
        print(f"  段落{i+1}[p{p['page']}]: {p['text'][:80]}...")
    
    # 打印第一张表格
    if result["tables"]:
        t = result["tables"][0]
        print(f"  表格1[p{t['page']}]: {t['title']}")
        print(f"  {t['markdown'][:200]}")
    
    # 3. 测试分块
    chunker = SemanticChunker()
    meta["file_path"] = pdf_path
    chunks = chunker.chunk(result["paragraphs"], result["tables"], meta)
    print(f"✓ 分块完成: {len(chunks)} 个 chunk")
    
    # 统计
    text_chunks = [c for c in chunks if c.metadata.chunk_type == "text"]
    table_chunks = [c for c in chunks if c.metadata.chunk_type == "table"]
    print(f"  - 文本块: {len(text_chunks)}, 表格块: {len(table_chunks)}")
    
    # 打印前 2 个 chunk 的元数据
    for i, c in enumerate(chunks[:2]):
        print(f"\n  Chunk {i+1}:")
        print(f"    ID: {c.metadata.chunk_id}")
        print(f"    章节: {c.metadata.section_title}")
        print(f"    类型: {c.metadata.chunk_type}")
        print(f"    Tokens: {c.token_count}")
        print(f"    内容预览: {c.content[:100]}...")
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True, help="测试用 PDF 文件路径")
    args = p.parse_args()
    test_single_file(args.pdf)
```

---

## 第九步：初始化 `requirements.txt`

在项目根目录创建 `requirements.txt`，按以下模板初始化，后续每安装一个包就追加到对应区块：

```
# =============================================================
# 特斯拉财报智能问答系统 · 依赖清单
# 安装命令：conda run -n tesla pip install -r requirements.txt
# =============================================================

# ── deepdoc / ragflow 核心依赖 ────────────────────────────────
# 运行 Step 0d 后，将 core/deepdoc/requirements.txt 的内容合并到此处
# （由 Claude Code 自动填写，勿手动猜测）

# ── 解析与分块模块 ────────────────────────────────────────────
tiktoken>=0.5.0          # token 数量估算，用于分块大小控制

# ── 向量数据库（下一阶段使用）────────────────────────────────
# chromadb>=0.4.0

# ── Embedding 模型（下一阶段使用）────────────────────────────
# sentence-transformers>=2.2.0

# ── API 服务（后续阶段使用）──────────────────────────────────
# fastapi>=0.100.0
# uvicorn>=0.23.0
```

**安装到 tesla 环境：**
```bash
conda run -n tesla pip install -r requirements.txt
```

---

## 执行顺序清单（Claude Code 请按此顺序执行）

```
【环境准备阶段】
□ Step 0a: conda env list | grep tesla                      ← 确认 tesla 环境存在
□ Step 0b: conda run -n tesla python --version             ← 确认 Python 版本
□ Step 0c: conda run -n tesla pip list                     ← 查看已安装包
□ Step 0d: find core/deepdoc -name "requirements*.txt"     ← 查找 deepdoc 依赖文件
□ Step 0e: 如有，运行 conda run -n tesla pip install -r core/deepdoc/requirements.txt
□ Step 0f: 将 deepdoc 依赖合并写入项目 requirements.txt

【探查 deepdoc 接口】
□ Step 1a: find core/deepdoc -type f -name "*.py" 查看目录结构
□ Step 1b: 阅读 deepdoc 核心解析类的源码，确认调用接口和返回格式

【创建项目文件】
□ Step 2:  创建 core/config.py
□ Step 3:  创建 core/parser/__init__.py（空文件）
□ Step 4:  创建 core/parser/metadata.py
□ Step 5:  创建 core/parser/pdf_parser.py（根据 Step 1 适配接口）
□ Step 6:  创建 core/parser/chunker.py
□ Step 7:  创建 core/utils/__init__.py（空文件）
□ Step 8:  创建 core/utils/file_utils.py
□ Step 9:  创建 core/parser/pipeline.py
□ Step 10: 创建 test_parser.py
□ Step 11: 创建 data/10-K/、data/10-Q/、output/、db_data/chroma/ 目录

【安装缺失依赖】
□ Step 12: conda run -n tesla pip install tiktoken          ← 安装 token 计数库
           → 同步追加到 requirements.txt
□ Step 13: 如运行中发现 ImportError，立即 conda run -n tesla pip install <包名>
           → 每次安装后都同步更新 requirements.txt

【验证与运行】
□ Step 14: conda run -n tesla python test_parser.py --pdf <任意一个测试PDF>
□ Step 15: 若测试通过，运行全量处理：
           conda run -n tesla python -m core.parser.pipeline
```

---

## 关键注意事项（Claude Code 必读）

1. **所有命令必须在 `tesla` conda 环境中运行**：使用 `conda run -n tesla <命令>` 格式，绝对不能直接用系统 `python` 或 `pip`
2. **每安装一个新包，立即同步到 `requirements.txt`**：带版本号，带注释说明用途，分类放置
3. **不要修改 `core/deepdoc/` 下任何文件**，只能调用，调用前必须先阅读其实际代码
4. **接口适配是最关键的步骤**：deepdoc 的返回格式在不同版本间可能不同，必须先读代码再写适配层
5. **表格的独立性是硬性要求**：每张表格必须是独立 chunk，不能和任何段落合并
6. **chunk_id 必须全局唯一**：格式为 `{fiscal_period}_p{page}_{index}`
7. **ChromaDB 元数据限制**：存入 ChromaDB 时，`section_hierarchy` 需要序列化为 JSON 字符串，`None` 值需转为 `-1`（见 `to_chroma_metadata()` 方法）
8. **文件命名规范是扫描的前提**：如果真实 PDF 文件名不符合规范，先在 `file_utils.py` 中添加对应的解析规则
```
