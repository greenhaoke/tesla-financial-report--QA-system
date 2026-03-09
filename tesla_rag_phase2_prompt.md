# Claude Code 执行提示词
## 特斯拉财报智能问答系统 · 第二阶段：检索增强与智能问答

---

## ⚠️ 环境规范（与第一阶段相同，所有命令必须遵守）

```bash
# ✅ 所有命令必须通过 conda run 在 tesla 环境中执行
conda run -n tesla python <脚本>
conda run -n tesla pip install <包名>

# ❌ 禁止直接调用系统环境
python <脚本>
pip install <包名>
```

每安装一个新包 → 立即追加到 `requirements.txt`（带版本号 + 用途注释）

---

## 设计说明（编写代码前必须理解）

### 一、已有数据结构回顾

ChromaDB 中存储的每条记录包含：
- `id`: chunk_id，如 `FY2023_p1_0`
- `embedding`: content 的向量表示
- `document`: content 原文
- `metadata`: 包含以下字段

```
source_file     → "tesla_10k_2023.pdf"
report_type     → "10-K" | "10-Q"
year            → 2023
quarter         → null（年报）或 1/2/3/4（季报）
fiscal_period   → "FY2023" | "2023Q2"
page_start      → 起始页（1-indexed）
page_end        → 结束页
section_title   → "Liquidity and Capital Resources"
section_hierarchy → JSON字符串，如 '["Item 7", "Liquidity"]'
chunk_type      → "text" | "table" | "mixed"
table_title     → 表格标题或 null
table_index     → 表格序号或 null
chunk_index     → 在文档内的全局序号
```

---

### 二、完整的检索架构设计

面对"2022年哪份季报提到供应链挑战，且该季度营收环比变化如何"这类复杂问题，
**单次向量检索无法解决**，需要一套完整的多阶段流程：

```
用户问题
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 1: 查询分析（Query Analyzer）       │
│  · 提取年份、季度、财务指标、关键词        │
│  · 判断问题类型（单点/对比/趋势/计算）     │
│  · 输出：结构化查询意图 + 检索计划         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 2: 并行混合检索（Hybrid Retriever） │
│  · 向量检索（语义相似）                   │
│  · BM25 关键词检索（精确术语匹配）        │
│  · 元数据预过滤（年份/季度/类型约束）     │
│  · 表格专项检索（数字类问题优先找表格）   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 3: 上下文扩展（Context Expander）  │
│  · Small2Big：按 section_title 扩展同节  │
│  · 相邻块合并：补充 chunk_index ±N 的块  │
│  · 去重 + 按页码重排序                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 4: 重排序（Reranker）              │
│  · 用 LLM 或 cross-encoder 对候选块打分 │
│  · 保留 top-K 最相关块                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 5: 答案生成（Answer Generator）    │
│  · 将上下文 + 问题送入 LLM              │
│  · 要求生成时引用来源（fiscal_period）   │
│  · 数字计算题：先提取数字再推理          │
└─────────────────────────────────────────┘
    │
    ▼
结构化答案（含来源引用 + 数据表格 + 推理链）
```

---

### 三、我额外补充的三个关键设计

**① 元数据预过滤（Pre-filtering）**
对于含有明确时间约束的问题（"2022年Q3"），应**先用元数据缩小搜索范围**，
再在子集内做向量检索。这比全量检索后过滤效率高 10 倍以上，且大幅提升精度。

```python
# 错误做法：全量检索，检索后过滤
results = collection.query(query_embeddings=..., n_results=100)
filtered = [r for r in results if r["year"] == 2022]

# 正确做法：ChromaDB where 子句预过滤，直接在子集上检索
results = collection.query(
    query_embeddings=...,
    n_results=20,
    where={"$and": [{"year": {"$eq": 2022}}, {"quarter": {"$eq": 3}}]}
)
```

**② 表格优先路由（Table-aware Routing）**
当问题包含"多少"、"增长率"、"环比"、"同比"、"毛利率"等数字型意图词时，
在向量检索的同时，额外触发一次 `chunk_type=table` 的专项检索，
保证财务数据表格不被纯文本段落"稀释"排名。

**③ 跨文档对比模式（Cross-doc Comparison）**
对于"对比 2021–2023 年"类问题，将查询拆解为多个子查询，
每个子查询只在对应 fiscal_period 内检索，最后并行合并结果，
避免某一年的信息过多导致另一年被截断。

---

## 工程目录规范

```
项目根目录/
├── app/                            # 前端（本阶段暂不涉及）
├── core/
│   ├── deepdoc/                    # 已有，不修改
│   ├── parser/                     # 第一阶段已完成
│   ├── retrieval/                  # 【本阶段新建】检索模块
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB 连接与查询封装
│   │   ├── bm25_retriever.py       # BM25 关键词检索
│   │   ├── hybrid_retriever.py     # 混合检索（向量 + BM25 融合）
│   │   ├── context_expander.py     # Small2Big 上下文扩展
│   │   └── reranker.py             # 检索结果重排序
│   ├── qa/                         # 【本阶段新建】问答模块
│   │   ├── __init__.py
│   │   ├── query_analyzer.py       # 查询意图分析与结构化
│   │   ├── retrieval_planner.py    # 根据意图生成检索计划
│   │   ├── answer_generator.py     # LLM 答案生成
│   │   └── pipeline.py             # 串联完整 QA 流程
│   ├── utils/
│   │   ├── file_utils.py           # 第一阶段已有
│   │   └── text_utils.py           # 【新增】文本处理工具
│   └── config.py                   # 补充检索相关配置
├── db_data/chroma/                 # 已有 ChromaDB 数据
├── output/                         # 已有 JSON 分块文件（用于 BM25 索引）
├── tests/
│   └── test_retrieval.py           # 检索验证脚本
└── requirements.txt                # 追加新依赖
```

---

## Step 0：环境检查与依赖安装

```bash
# 确认环境
conda run -n tesla python --version
conda run -n tesla pip list | grep -E "chromadb|rank|bm25|openai|anthropic"

# 安装本阶段新增依赖（逐一安装，每次安装后追加到 requirements.txt）
conda run -n tesla pip install rank-bm25          # BM25 检索
conda run -n tesla pip install anthropic          # Claude API（答案生成）

# 查询安装版本后写入 requirements.txt
conda run -n tesla pip show rank-bm25 | grep Version
conda run -n tesla pip show anthropic | grep Version
```

安装完成后，在 `requirements.txt` 中追加（带注释）：
```
# ── 检索模块 ─────────────────────────────────────────────
rank-bm25>=0.2.2         # BM25 关键词检索，用于混合检索中的精确匹配

# ── LLM 调用 ─────────────────────────────────────────────
anthropic>=0.30.0        # Claude API，用于查询分析和答案生成
```

---

## Step 1：补充 `core/config.py`

在已有配置末尾追加检索相关配置：

```python
# core/config.py（追加以下内容）

import os

# ── LLM 配置 ──────────────────────────────────────────────
LLM_CONFIG = {
    "provider": "anthropic",           # 使用 Claude
    "model": "claude-opus-4-5",        # 主力模型（复杂推理）
    "fast_model": "claude-haiku-4-5-20251001",  # 轻量模型（意图分析等简单任务）
    "api_key_env": "ANTHROPIC_API_KEY", # 从环境变量读取
    "max_tokens": 4096,
    "temperature": 0.1,                # 财务问答要求精确，低温度
}

# ── 检索配置 ───────────────────────────────────────────────
RETRIEVAL_CONFIG = {
    # ChromaDB
    "chroma_collection_name": "tesla_financial_reports",
    
    # 混合检索权重（vector_weight + bm25_weight = 1.0）
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    
    # 初始检索数量（Hybrid 合并前每路各取多少）
    "vector_top_k": 20,
    "bm25_top_k": 20,
    
    # 重排后保留的最终 chunk 数量
    "final_top_k": 8,
    
    # Small2Big 扩展配置
    "expand_by_section": True,         # 按 section_title 扩展
    "expand_neighbor_n": 1,            # 同时取相邻 N 个 chunk
    "max_expanded_chunks": 20,         # 扩展后最多保留的 chunk 数
    
    # 表格专项检索
    "table_search_enabled": True,
    "table_top_k": 5,                  # 额外检索的表格数量
    
    # 数字意图关键词（触发表格优先检索）
    "numeric_intent_keywords": [
        "多少", "增长", "下降", "环比", "同比", "毛利率", "营收",
        "利润", "现金流", "revenue", "margin", "growth", "increase",
        "decrease", "quarter-over-quarter", "year-over-year", "QoQ", "YoY",
        "free cash flow", "FCF", "EPS", "gross profit"
    ],
}

# ── BM25 索引文件路径 ──────────────────────────────────────
BM25_INDEX_PATH = ROOT_DIR / "db_data" / "bm25_index.pkl"
```

---

## Step 2：实现 `core/retrieval/vector_store.py`

```python
# core/retrieval/vector_store.py
"""
ChromaDB 连接封装，提供带元数据过滤的向量检索接口。
"""
import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb import Collection

from core.config import DB_DATA_DIR, RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 向量存储封装。"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_DATA_DIR))
        self.collection: Collection = self.client.get_collection(
            name=RETRIEVAL_CONFIG["chroma_collection_name"]
        )
        logger.info(
            f"VectorStore 已连接，集合大小: {self.collection.count()} 条"
        )

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 20,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似性检索，支持元数据预过滤。

        Args:
            query_embedding: 查询向量
            n_results: 返回数量
            where: ChromaDB 元数据过滤条件，如：
                   {"year": {"$eq": 2022}}
                   {"$and": [{"year": {"$eq": 2022}}, {"quarter": {"$eq": 3}}]}
            where_document: 文档内容过滤（全文搜索，较慢）

        Returns:
            标准化的 chunk 字典列表，每个包含 id/content/metadata/score
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        raw = self.collection.query(**kwargs)

        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "content": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                # ChromaDB 返回的是 L2 距离，转换为 [0,1] 相似度分数
                "score": 1.0 / (1.0 + raw["distances"][0][i]),
                "source": "vector",
            })
        return results

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """按 chunk_id 批量获取，用于 Small2Big 扩展。"""
        raw = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        results = []
        for i in range(len(raw["ids"])):
            results.append({
                "id": raw["ids"][i],
                "content": raw["documents"][i],
                "metadata": raw["metadatas"][i],
                "score": 1.0,  # 直接获取，赋满分
                "source": "expansion",
            })
        return results

    def get_by_metadata_filter(
        self,
        where: Dict,
        chunk_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        纯元数据过滤查询（不使用向量），用于表格专项检索和精确定位。
        
        例：获取 2022Q3 的所有表格 chunk
        where = {"$and": [{"year": {"$eq": 2022}}, {"quarter": {"$eq": 3}}]}
        chunk_type = "table"
        """
        actual_where = dict(where)
        if chunk_type:
            type_filter = {"chunk_type": {"$eq": chunk_type}}
            if "$and" in actual_where:
                actual_where["$and"].append(type_filter)
            else:
                actual_where = {"$and": [actual_where, type_filter]}

        raw = self.collection.get(
            where=actual_where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        results = []
        for i in range(len(raw["ids"])):
            results.append({
                "id": raw["ids"][i],
                "content": raw["documents"][i],
                "metadata": raw["metadatas"][i],
                "score": 0.9,
                "source": "metadata_filter",
            })
        return results

    def get_all_for_bm25(self) -> List[Dict[str, Any]]:
        """获取全量数据，用于构建 BM25 索引。"""
        raw = self.collection.get(include=["documents", "metadatas"])
        results = []
        for i in range(len(raw["ids"])):
            results.append({
                "id": raw["ids"][i],
                "content": raw["documents"][i],
                "metadata": raw["metadatas"][i],
            })
        logger.info(f"加载全量数据用于 BM25: {len(results)} 条")
        return results
```

---

## Step 3：实现 `core/retrieval/bm25_retriever.py`

```python
# core/retrieval/bm25_retriever.py
"""
BM25 关键词检索器。
首次运行时从 ChromaDB 加载全量数据构建索引，并序列化缓存到磁盘。
"""
import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from core.config import BM25_INDEX_PATH, RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)

# 金融术语不拆分的特殊词组（保留完整形式）
FINANCIAL_PHRASES = [
    "free cash flow", "gross margin", "operating margin", "net income",
    "earnings per share", "revenue recognition", "supply chain",
    "capital expenditure", "research and development", "r&d",
    "accounts receivable", "accounts payable", "automotive revenue",
    "energy generation", "services revenue", "quarter-over-quarter",
    "year-over-year", "10-k", "10-q",
]


def tokenize(text: str) -> List[str]:
    """
    财务文本分词：
    1. 保留金融术语短语
    2. 保留数字+单位（如 $1.2B, 23.5%）
    3. 小写化，移除无意义标点
    """
    text_lower = text.lower()

    # 用占位符保护金融短语
    placeholders = {}
    for phrase in FINANCIAL_PHRASES:
        if phrase in text_lower:
            placeholder = f"__PHRASE_{len(placeholders)}__"
            placeholders[placeholder] = phrase.replace(" ", "_")
            text_lower = text_lower.replace(phrase, placeholder)

    # 保留数字相关 token（如 $1.2b, 23.5%, q3, fy2022）
    tokens = re.findall(
        r'\$[\d,.]+[bm]?\b'   # 金额
        r'|\d+\.?\d*%'         # 百分比
        r'|q[1-4]\b'           # 季度
        r'|fy\d{4}\b'          # 财年
        r'|\d{4}(?:q[1-4])?\b' # 年份/年季
        r'|[a-z__]+',           # 普通词和占位符
        text_lower
    )

    # 还原占位符
    result = []
    for token in tokens:
        if token in placeholders:
            result.append(placeholders[token])
        elif len(token) > 1:  # 过滤单字符噪声
            result.append(token)

    return result


class BM25Retriever:
    """BM25 关键词检索器，支持元数据过滤。"""

    def __init__(self, vector_store=None):
        self.corpus: List[Dict[str, Any]] = []  # 原始 chunk 数据
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._vector_store = vector_store

        # 优先从磁盘加载缓存索引
        if BM25_INDEX_PATH.exists():
            self._load_index()
        elif vector_store is not None:
            self.build_index(vector_store.get_all_for_bm25())
        else:
            logger.warning("BM25 索引未加载，请先调用 build_index()")

    def build_index(self, chunks: List[Dict[str, Any]]):
        """从 chunk 列表构建 BM25 索引并缓存到磁盘。"""
        logger.info(f"构建 BM25 索引，共 {len(chunks)} 条...")
        self.corpus = chunks
        self.tokenized_corpus = [tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
        logger.info("BM25 索引构建完成并已缓存")

    def search(
        self,
        query: str,
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索，支持 where 过滤（在 BM25 结果上做后过滤）。
        
        注意：BM25 不原生支持预过滤，通过后过滤实现。
        如果 where 过滤掉大量结果，建议适当放大初始检索数量。
        """
        if self.bm25 is None:
            logger.error("BM25 索引未初始化")
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 按分数排序，取 top n*5（留余量供过滤）
        n_fetch = min(n_results * 5, len(self.corpus))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_fetch]

        results = []
        for idx in top_indices:
            chunk = self.corpus[idx]
            if where and not self._match_where(chunk["metadata"], where):
                continue
            results.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": float(scores[idx]),
                "source": "bm25",
            })
            if len(results) >= n_results:
                break

        return results

    def _match_where(self, metadata: Dict, where: Dict) -> bool:
        """简单的 where 条件匹配（与 ChromaDB where 语法对齐）。"""
        if "$and" in where:
            return all(self._match_where(metadata, cond) for cond in where["$and"])
        if "$or" in where:
            return any(self._match_where(metadata, cond) for cond in where["$or"])
        for field, condition in where.items():
            val = metadata.get(field)
            if isinstance(condition, dict):
                op, target = next(iter(condition.items()))
                if op == "$eq" and val != target:
                    return False
                elif op == "$ne" and val == target:
                    return False
                elif op == "$in" and val not in target:
                    return False
                elif op == "$gte" and (val is None or val < target):
                    return False
                elif op == "$lte" and (val is None or val > target):
                    return False
            else:
                if val != condition:
                    return False
        return True

    def _save_index(self):
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"corpus": self.corpus, "tokenized": self.tokenized_corpus}, f)
        logger.info(f"BM25 索引已保存: {BM25_INDEX_PATH}")

    def _load_index(self):
        logger.info(f"加载 BM25 索引: {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.tokenized_corpus = data["tokenized"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 索引加载完成: {len(self.corpus)} 条")
```

---

## Step 4：实现 `core/retrieval/hybrid_retriever.py`

```python
# core/retrieval/hybrid_retriever.py
"""
混合检索：向量检索（语义）+ BM25（关键词）→ RRF 融合排序

RRF（Reciprocal Rank Fusion）：一种简单有效的多路检索融合算法，
不需要对两路分数做归一化，只依赖排名位置。
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from core.retrieval.vector_store import VectorStore
from core.retrieval.bm25_retriever import BM25Retriever
from core.config import RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)


def rrf_fusion(
    result_lists: List[List[Dict]],
    weights: List[float],
    k: int = 60,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion 多路结果融合。
    
    公式：score(d) = Σ weight_i / (k + rank_i(d))
    k=60 是经验最优值，防止头部排名差异过大。
    """
    doc_scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict] = {}

    for result_list, weight in zip(result_lists, weights):
        for rank, doc in enumerate(result_list):
            doc_id = doc["id"]
            rrf_score = weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

    # 合并融合分数
    fused = []
    for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(doc_data[doc_id])
        item["rrf_score"] = score
        item["score"] = score  # 统一字段名
        fused.append(item)

    return fused


class HybridRetriever:
    """
    混合检索器：向量 + BM25 + 表格专项 → RRF 融合。

    支持：
    - 元数据预过滤（年份、季度、报告类型约束）
    - 数字意图自动触发表格专项检索
    - 跨文档并行检索（对比类问题）
    """

    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25Retriever, embed_fn):
        """
        Args:
            vector_store: VectorStore 实例
            bm25_retriever: BM25Retriever 实例
            embed_fn: 将文本转为向量的函数（与第一阶段使用相同的 embedding 模型）
        """
        self.vs = vector_store
        self.bm25 = bm25_retriever
        self.embed_fn = embed_fn
        self.cfg = RETRIEVAL_CONFIG

    def retrieve(
        self,
        query: str,
        n_results: int = None,
        where: Optional[Dict] = None,
        force_table_search: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        主检索入口。

        Args:
            query: 用户查询文本
            n_results: 最终返回数量（默认用 config 中的 final_top_k）
            where: 元数据过滤条件（由 QueryAnalyzer 生成）
            force_table_search: 强制触发表格专项检索
        """
        n_results = n_results or self.cfg["final_top_k"]

        # 1. 生成查询向量
        query_vec = self.embed_fn(query)

        # 2. 向量检索
        vec_results = self.vs.query(
            query_embedding=query_vec,
            n_results=self.cfg["vector_top_k"],
            where=where,
        )

        # 3. BM25 检索
        bm25_results = self.bm25.search(
            query=query,
            n_results=self.cfg["bm25_top_k"],
            where=where,
        )

        # 4. 判断是否需要表格专项检索
        needs_table = force_table_search or self._has_numeric_intent(query)
        table_results = []
        if needs_table and self.cfg["table_search_enabled"] and where:
            # 用元数据过滤直接捞出相关表格 chunk
            table_where = dict(where)
            table_results = self.vs.get_by_metadata_filter(
                where=table_where,
                chunk_type="table",
                limit=self.cfg["table_top_k"],
            )
            logger.info(f"表格专项检索: 找到 {len(table_results)} 张表格")

        # 5. RRF 融合
        result_lists = [vec_results, bm25_results]
        weights = [self.cfg["vector_weight"], self.cfg["bm25_weight"]]

        if table_results:
            result_lists.append(table_results)
            weights.append(0.3)  # 表格结果额外加权

        fused = rrf_fusion(result_lists, weights)

        logger.info(
            f"混合检索完成: 向量{len(vec_results)} + BM25{len(bm25_results)}"
            f" + 表格{len(table_results)} → 融合后{len(fused)}，取前{n_results}"
        )
        return fused[:n_results * 3]  # 返回更多，供后续扩展和重排

    def retrieve_multi_period(
        self,
        query: str,
        fiscal_periods: List[str],
        n_per_period: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        跨文档并行检索，用于对比类问题。
        
        对每个 fiscal_period 独立检索，返回 {period: [chunks]} 字典。
        避免某一年信息过多挤占其他年份的配额。
        """
        results_by_period = {}
        query_vec = self.embed_fn(query)

        for period in fiscal_periods:
            # 根据 period 格式判断是年报还是季报
            if period.startswith("FY"):
                year = int(period[2:])
                where = {"$and": [
                    {"year": {"$eq": year}},
                    {"report_type": {"$eq": "10-K"}},
                ]}
            else:
                # 格式：2022Q3
                year = int(period[:4])
                quarter = int(period[5])
                where = {"$and": [
                    {"year": {"$eq": year}},
                    {"quarter": {"$eq": quarter}},
                ]}

            vec_results = self.vs.query(
                query_embedding=query_vec,
                n_results=n_per_period,
                where=where,
            )
            bm25_results = self.bm25.search(
                query=query, n_results=n_per_period, where=where
            )
            fused = rrf_fusion([vec_results, bm25_results], [0.6, 0.4])
            results_by_period[period] = fused[:n_per_period]
            logger.info(f"[{period}] 检索到 {len(fused)} 条结果")

        return results_by_period

    def _has_numeric_intent(self, query: str) -> bool:
        """判断问题是否包含数字类意图（需要表格支持）。"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.cfg["numeric_intent_keywords"])
```

---

## Step 5：实现 `core/retrieval/context_expander.py`

```python
# core/retrieval/context_expander.py
"""
上下文扩展：Small2Big 策略实现。

核心思路：
- 向量检索找到的是精准的小块（small）
- 但理解需要完整章节上下文（big）
- 通过 section_title 和 chunk_index 扩展找到"大块"
"""
import json
import logging
from typing import List, Dict, Any, Set

from core.retrieval.vector_store import VectorStore
from core.config import RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)


class ContextExpander:
    """基于 section_title 和相邻 chunk 的上下文扩展器。"""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.cfg = RETRIEVAL_CONFIG

    def expand(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        扩展检索结果的上下文。

        扩展策略（按优先级）：
        1. 同 section：找出 same (fiscal_period + section_title) 的所有 chunk
        2. 相邻 chunk：根据 chunk_index ± N 找前后块
        3. 去重合并，按 (fiscal_period, page_start, chunk_index) 排序
        """
        if not chunks:
            return chunks

        expanded_ids: Set[str] = set()
        expanded_chunks: List[Dict] = []

        for chunk in chunks:
            # 原始 chunk 始终保留
            if chunk["id"] not in expanded_ids:
                expanded_ids.add(chunk["id"])
                expanded_chunks.append(chunk)

            meta = chunk["metadata"]

            # 策略1：按 section_title 扩展（同文档同章节）
            if self.cfg["expand_by_section"] and meta.get("section_title"):
                section_chunks = self._get_same_section(
                    fiscal_period=meta["fiscal_period"],
                    section_title=meta["section_title"],
                )
                for sc in section_chunks:
                    if sc["id"] not in expanded_ids:
                        expanded_ids.add(sc["id"])
                        expanded_chunks.append(sc)

            # 策略2：相邻 chunk 扩展
            n = self.cfg["expand_neighbor_n"]
            if n > 0:
                neighbor_chunks = self._get_neighbors(
                    fiscal_period=meta["fiscal_period"],
                    chunk_index=meta["chunk_index"],
                    n=n,
                )
                for nc in neighbor_chunks:
                    if nc["id"] not in expanded_ids:
                        expanded_ids.add(nc["id"])
                        expanded_chunks.append(nc)

        # 限制总数，避免上下文过长
        max_chunks = self.cfg["max_expanded_chunks"]
        if len(expanded_chunks) > max_chunks:
            # 优先保留原始检索结果，其次是扩展内容
            original_ids = {c["id"] for c in chunks}
            originals = [c for c in expanded_chunks if c["id"] in original_ids]
            extras = [c for c in expanded_chunks if c["id"] not in original_ids]
            expanded_chunks = originals + extras[: max_chunks - len(originals)]

        # 按来源文档和页码排序，保证上下文连贯
        expanded_chunks.sort(key=lambda c: (
            c["metadata"].get("fiscal_period", ""),
            c["metadata"].get("page_start", 0),
            c["metadata"].get("chunk_index", 0),
        ))

        logger.info(
            f"上下文扩展: {len(chunks)} → {len(expanded_chunks)} 个 chunk"
        )
        return expanded_chunks

    def _get_same_section(
        self, fiscal_period: str, section_title: str
    ) -> List[Dict]:
        """获取同一财报同一章节的所有 chunk。"""
        # 从 fiscal_period 解析年份和季度
        year, quarter = self._parse_fiscal_period(fiscal_period)
        where = self._build_period_where(year, quarter)

        # ChromaDB 不支持 string 的 contains 过滤，
        # 用精确匹配 section_title
        if "$and" in where:
            where["$and"].append({"section_title": {"$eq": section_title}})
        else:
            where = {"$and": [where, {"section_title": {"$eq": section_title}}]}

        return self.vs.get_by_metadata_filter(where=where, limit=30)

    def _get_neighbors(
        self, fiscal_period: str, chunk_index: int, n: int
    ) -> List[Dict]:
        """获取相邻 N 个 chunk（通过 chunk_index 范围过滤）。"""
        year, quarter = self._parse_fiscal_period(fiscal_period)
        where = self._build_period_where(year, quarter)

        target_indices = list(range(
            max(0, chunk_index - n), chunk_index
        )) + list(range(chunk_index + 1, chunk_index + n + 1))

        results = []
        for idx in target_indices:
            idx_where = dict(where)
            if "$and" in idx_where:
                idx_where["$and"].append({"chunk_index": {"$eq": idx}})
            else:
                idx_where = {"$and": [idx_where, {"chunk_index": {"$eq": idx}}]}
            chunks = self.vs.get_by_metadata_filter(where=idx_where, limit=1)
            results.extend(chunks)
        return results

    def _parse_fiscal_period(self, fiscal_period: str):
        """解析 fiscal_period，返回 (year, quarter)。"""
        if fiscal_period.startswith("FY"):
            return int(fiscal_period[2:]), None
        else:
            year = int(fiscal_period[:4])
            quarter = int(fiscal_period[5]) if len(fiscal_period) > 4 else None
            return year, quarter

    def _build_period_where(self, year: int, quarter) -> Dict:
        """根据年份和季度构建 ChromaDB where 条件。"""
        if quarter is None:
            return {"$and": [
                {"year": {"$eq": year}},
                {"report_type": {"$eq": "10-K"}},
            ]}
        else:
            return {"$and": [
                {"year": {"$eq": year}},
                {"quarter": {"$eq": quarter}},
            ]}
```

---

## Step 6：实现 `core/qa/query_analyzer.py`

```python
# core/qa/query_analyzer.py
"""
查询意图分析器：使用 LLM 将自然语言问题解析为结构化检索意图。

输出的结构化意图将直接驱动 HybridRetriever 的检索参数。
"""
import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import anthropic

from core.config import LLM_CONFIG

logger = logging.getLogger(__name__)

QUERY_ANALYSIS_PROMPT = """你是一个专业的金融文档检索助手。请分析以下问题，提取检索所需的结构化信息。

问题：{question}

请以 JSON 格式返回分析结果，字段说明如下：
- question_type: 问题类型，必须是以下之一：
  * "single_point"    → 询问某个具体事实（如"2022年Q3营收是多少"）
  * "comparison"      → 跨时间/文档对比（如"对比2021-2023年毛利率"）
  * "trend"           → 趋势分析（如"自2020年以来的增长趋势"）
  * "calculation"     → 需要数学计算（如"环比增长率"）
  * "multi_step"      → 需要多步推理（如"哪个季度提到了X，且该季度Y变化如何"）
- years: 涉及的年份列表，如 [2022, 2023]，不确定则为 []
- quarters: 涉及的季度列表，如 [3]，年报问题则为 []
- report_types: 涉及的报告类型列表，可含 "10-K", "10-Q"，不限制则为 []
- fiscal_periods: 涉及的财务期间标签，如 ["FY2022", "2022Q3"]，用于跨文档对比检索
- financial_metrics: 涉及的财务指标关键词列表，如 ["automotive gross margin", "revenue"]
- intent_keywords: 检索关键词列表（最重要的 3-5 个词组），如 ["supply chain", "gross margin"]
- needs_table: 布尔值，问题是否需要表格数据（含数字计算、财务数据对比时为 true）
- needs_text: 布尔值，问题是否需要文本叙述（含"管理层讨论"、"原因"、"背景"时为 true）
- sub_questions: 如果是 multi_step 类型，拆解为子问题列表；其他类型为 []
- metadata_filter: ChromaDB where 过滤条件（JSON格式），没有约束则为 null

示例输出：
{{
  "question_type": "multi_step",
  "years": [2022],
  "quarters": [],
  "report_types": ["10-Q"],
  "fiscal_periods": ["2022Q1", "2022Q2", "2022Q3"],
  "financial_metrics": ["revenue", "quarter-over-quarter"],
  "intent_keywords": ["supply chain", "challenges", "revenue"],
  "needs_table": true,
  "needs_text": true,
  "sub_questions": [
    "2022年哪份季报提到了供应链挑战？",
    "该季度的营收是多少？",
    "上一季度的营收是多少？",
    "计算环比变化率"
  ],
  "metadata_filter": {{"$and": [{{"year": {{"$eq": 2022}}}}, {{"report_type": {{"$eq": "10-Q"}}}}]}}
}}

只返回 JSON，不要有其他文字。"""


@dataclass
class QueryIntent:
    question_type: str
    years: List[int]
    quarters: List[int]
    report_types: List[str]
    fiscal_periods: List[str]
    financial_metrics: List[str]
    intent_keywords: List[str]
    needs_table: bool
    needs_text: bool
    sub_questions: List[str]
    metadata_filter: Optional[Dict]
    raw_question: str


class QueryAnalyzer:
    """使用 Claude 分析查询意图。"""

    def __init__(self):
        import os
        self.client = anthropic.Anthropic(
            api_key=os.environ.get(LLM_CONFIG["api_key_env"])
        )
        self.fast_model = LLM_CONFIG["fast_model"]

    def analyze(self, question: str) -> QueryIntent:
        """分析问题，返回结构化意图。"""
        logger.info(f"分析查询意图: {question[:60]}...")

        prompt = QUERY_ANALYSIS_PROMPT.format(question=question)

        response = self.client.messages.create(
            model=self.fast_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_json = response.content[0].text.strip()

        # 提取 JSON（防止 LLM 包裹额外文字）
        match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        if match:
            raw_json = match.group()

        data = json.loads(raw_json)
        intent = QueryIntent(
            question_type=data.get("question_type", "single_point"),
            years=data.get("years", []),
            quarters=data.get("quarters", []),
            report_types=data.get("report_types", []),
            fiscal_periods=data.get("fiscal_periods", []),
            financial_metrics=data.get("financial_metrics", []),
            intent_keywords=data.get("intent_keywords", []),
            needs_table=data.get("needs_table", False),
            needs_text=data.get("needs_text", True),
            sub_questions=data.get("sub_questions", []),
            metadata_filter=data.get("metadata_filter"),
            raw_question=question,
        )

        logger.info(
            f"意图分析完成: type={intent.question_type}, "
            f"years={intent.years}, periods={intent.fiscal_periods}"
        )
        return intent
```

---

## Step 7：实现 `core/qa/answer_generator.py`

```python
# core/qa/answer_generator.py
"""
答案生成器：将检索到的上下文 + 问题送入 Claude 生成最终答案。

支持：
- 带来源引用的结构化答案
- 数字计算的分步推理
- 对比类问题的表格化输出
"""
import os
import json
import logging
from typing import List, Dict, Any
import anthropic

from core.config import LLM_CONFIG

logger = logging.getLogger(__name__)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """将 chunk 列表格式化为 LLM 可读的上下文字符串。"""
    parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        period = meta.get("fiscal_period", "Unknown")
        section = meta.get("section_title", "")
        chunk_type = meta.get("chunk_type", "text")
        page = meta.get("page_start", "?")

        header = f"[来源 {i+1}] {period} | {section} | 第{page}页"
        if chunk_type == "table":
            table_title = meta.get("table_title", "")
            header += f" | 表格: {table_title}"

        parts.append(f"{header}\n{chunk['content']}")

    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """你是一位专业的特斯拉财报分析师，拥有深厚的财务知识。
你的任务是基于提供的财报原文片段，准确、客观地回答关于特斯拉财务状况的问题。

回答规范：
1. **严格基于原文**：只使用提供的原文片段中的信息，不要凭记忆补充未提供的数据
2. **明确引用来源**：每个关键数据或结论后，用 [来源N: 期间 | 章节] 格式注明出处
3. **数字计算要展示过程**：如需计算环比/同比，写出计算公式和中间步骤
4. **区分文本和数据**：财务数据来自表格时说明，文字描述来自 MD&A 等章节时说明
5. **信息不足时明确说明**：如果提供的片段不足以回答某部分问题，明确指出"原文中未找到相关信息"
6. **结构化输出**：复杂问题用分点回答，对比问题优先用 Markdown 表格呈现"""

ANSWER_PROMPT = """请基于以下特斯拉财报原文片段，回答用户的问题。

## 财报原文片段

{context}

## 用户问题

{question}

## 回答要求

{special_instructions}

请现在给出详细、准确的分析答案："""


def _get_special_instructions(question_type: str, needs_table: bool) -> str:
    instructions = {
        "single_point": "直接给出答案，注明数据来源的期间和章节。",
        "comparison": "用 Markdown 表格对比不同期间的数据，表格后加文字分析。",
        "trend": "按时间顺序列出数据点，分析趋势方向和原因。",
        "calculation": "明确写出计算公式、代入数值、得出结果，如有单位换算请注明。",
        "multi_step": "按子问题逐步回答，最后综合得出结论。",
    }
    base = instructions.get(question_type, "详细回答问题，注明来源。")
    if needs_table:
        base += "\n注意：优先使用表格数据中的精确数字，文本中的描述作为补充。"
    return base


class AnswerGenerator:
    """使用 Claude 生成最终答案。"""

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ.get(LLM_CONFIG["api_key_env"])
        )
        self.model = LLM_CONFIG["model"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.temperature = LLM_CONFIG["temperature"]

    def generate(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        question_type: str = "single_point",
        needs_table: bool = False,
    ) -> Dict[str, Any]:
        """
        生成答案。

        Returns:
            {
                "answer": str,           # 最终答案文本
                "sources": list,         # 引用的来源列表
                "model": str,            # 使用的模型
                "context_count": int,    # 使用的 chunk 数量
            }
        """
        if not context_chunks:
            return {
                "answer": "抱歉，在财报中未找到与该问题相关的内容。请确认查询的时间范围和关键词是否正确。",
                "sources": [],
                "model": self.model,
                "context_count": 0,
            }

        context_str = _format_context(context_chunks)
        special_inst = _get_special_instructions(question_type, needs_table)

        prompt = ANSWER_PROMPT.format(
            context=context_str,
            question=question,
            special_instructions=special_inst,
        )

        logger.info(f"生成答案，使用 {len(context_chunks)} 个 chunk，模型: {self.model}")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = response.content[0].text

        # 提取引用来源
        sources = list({
            f"{c['metadata'].get('fiscal_period')} | {c['metadata'].get('section_title')}"
            for c in context_chunks
        })

        return {
            "answer": answer_text,
            "sources": sorted(sources),
            "model": self.model,
            "context_count": len(context_chunks),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        }
```

---

## Step 8：实现 `core/qa/pipeline.py`（完整 QA 流程串联）

```python
# core/qa/pipeline.py
"""
完整问答流程串联：
QueryAnalyzer → RetrievalPlanner → HybridRetriever → ContextExpander → AnswerGenerator
"""
import os
import logging
from typing import Optional, Callable, List, Dict, Any

from core.qa.query_analyzer import QueryAnalyzer, QueryIntent
from core.retrieval.vector_store import VectorStore
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.context_expander import ContextExpander
from core.qa.answer_generator import AnswerGenerator
from core.config import RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)


class TeslaQAPipeline:
    """
    特斯拉财报智能问答系统主入口。
    
    使用方式：
        pipeline = TeslaQAPipeline(embed_fn=your_embed_function)
        result = pipeline.ask("2022年Q3的汽车毛利率是多少？")
        print(result["answer"])
    """

    def __init__(self, embed_fn: Callable[[str], List[float]]):
        """
        Args:
            embed_fn: 将文本转为向量的函数。
                      必须与第一阶段入库时使用的 embedding 模型相同！
        """
        logger.info("初始化 TeslaQAPipeline...")

        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever(vector_store=self.vector_store)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25,
            embed_fn=embed_fn,
        )
        self.expander = ContextExpander(vector_store=self.vector_store)
        self.analyzer = QueryAnalyzer()
        self.generator = AnswerGenerator()

        logger.info("TeslaQAPipeline 初始化完成")

    def ask(
        self,
        question: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        主问答接口。
        
        Returns:
            {
                "question": str,
                "answer": str,
                "sources": list,
                "intent": dict,        # 解析出的查询意图
                "retrieved_chunks": list,  # 检索到的原始 chunk（调试用）
                "context_count": int,
                "usage": dict,
            }
        """
        logger.info(f"收到问题: {question}")

        # ── Step 1: 查询意图分析 ─────────────────────────────────
        intent = self.analyzer.analyze(question)

        if verbose:
            logger.info(f"意图: {intent.__dict__}")

        # ── Step 2: 根据意图类型选择检索策略 ─────────────────────
        if intent.question_type == "comparison" and len(intent.fiscal_periods) > 1:
            # 跨文档对比：并行检索各期间
            chunks = self._retrieve_comparison(question, intent)
        elif intent.question_type == "multi_step" and intent.sub_questions:
            # 多步推理：每个子问题独立检索
            chunks = self._retrieve_multi_step(question, intent)
        else:
            # 单点/趋势/计算：单次混合检索
            chunks = self._retrieve_single(question, intent)

        if verbose:
            logger.info(f"初始检索: {len(chunks)} 个 chunk")

        # ── Step 3: 上下文扩展（Small2Big）──────────────────────
        expanded_chunks = self.expander.expand(chunks)

        if verbose:
            logger.info(f"扩展后: {len(expanded_chunks)} 个 chunk")

        # ── Step 4: 最终截断（控制上下文长度）────────────────────
        final_chunks = expanded_chunks[:RETRIEVAL_CONFIG["max_expanded_chunks"]]

        # ── Step 5: 答案生成 ──────────────────────────────────────
        result = self.generator.generate(
            question=question,
            context_chunks=final_chunks,
            question_type=intent.question_type,
            needs_table=intent.needs_table,
        )

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": {
                "type": intent.question_type,
                "years": intent.years,
                "quarters": intent.quarters,
                "fiscal_periods": intent.fiscal_periods,
            },
            "retrieved_chunks": final_chunks if verbose else [],
            "context_count": result["context_count"],
            "usage": result.get("usage", {}),
        }

    def _retrieve_single(
        self, question: str, intent: QueryIntent
    ) -> List[Dict]:
        """单次混合检索。"""
        return self.retriever.retrieve(
            query=question,
            where=intent.metadata_filter,
            force_table_search=intent.needs_table,
        )

    def _retrieve_comparison(
        self, question: str, intent: QueryIntent
    ) -> List[Dict]:
        """跨文档并行检索，合并结果。"""
        results_by_period = self.retriever.retrieve_multi_period(
            query=question,
            fiscal_periods=intent.fiscal_periods,
            n_per_period=RETRIEVAL_CONFIG["final_top_k"] // max(len(intent.fiscal_periods), 1),
        )
        # 合并所有期间的结果
        all_chunks = []
        for chunks in results_by_period.values():
            all_chunks.extend(chunks)
        return all_chunks

    def _retrieve_multi_step(
        self, question: str, intent: QueryIntent
    ) -> List[Dict]:
        """多步推理：对每个子问题检索，合并去重。"""
        seen_ids = set()
        all_chunks = []

        for sub_q in intent.sub_questions:
            sub_chunks = self.retriever.retrieve(
                query=sub_q,
                where=intent.metadata_filter,
                force_table_search=intent.needs_table,
            )
            for chunk in sub_chunks:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_chunks.append(chunk)

        return all_chunks
```

---

## Step 9：编写测试脚本 `tests/test_retrieval.py`

```python
# tests/test_retrieval.py
"""
检索流程端到端测试。
用法：conda run -n tesla python tests/test_retrieval.py
"""
import sys
import os

# 注意：替换为你的 embedding 函数
# 必须和第一阶段入库时使用的完全相同
def get_embed_fn():
    """
    返回 embedding 函数。
    请根据第一阶段实际使用的 embedding 模型修改此函数。
    
    示例（使用 sentence-transformers）：
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        return lambda text: model.encode(text).tolist()
    
    示例（使用 OpenAI）：
        from openai import OpenAI
        client = OpenAI()
        def embed(text):
            resp = client.embeddings.create(model="text-embedding-3-small", input=text)
            return resp.data[0].embedding
        return embed
    """
    raise NotImplementedError("请实现 get_embed_fn()，使用与入库时相同的 embedding 模型！")


TEST_QUESTIONS = [
    # 单点问题（测试精确检索）
    {
        "q": "特斯拉2022年全年的汽车营收是多少？",
        "type": "single_point",
        "expect_period": "FY2022",
    },
    # 关键词精确匹配（测试 BM25）
    {
        "q": "2022年哪些季报中提到了供应链挑战（supply chain challenges）？",
        "type": "keyword_match",
        "expect_keyword": "supply chain",
    },
    # 对比问题（测试跨文档检索）
    {
        "q": "对比2021年、2022年、2023年特斯拉的汽车毛利率变化趋势",
        "type": "comparison",
        "expect_periods": ["FY2021", "FY2022", "FY2023"],
    },
    # 多步推理（测试子问题分解）
    {
        "q": "2022年哪份季报提到了供应链挑战？并计算该季度的营收环比变化率",
        "type": "multi_step",
    },
]


def run_tests():
    embed_fn = get_embed_fn()

    from core.qa.pipeline import TeslaQAPipeline
    pipeline = TeslaQAPipeline(embed_fn=embed_fn)

    for i, test in enumerate(TEST_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(TEST_QUESTIONS)}: {test['type']}")
        print(f"问题: {test['q']}")
        print("-" * 60)

        result = pipeline.ask(test["q"], verbose=True)

        print(f"意图类型: {result['intent']['type']}")
        print(f"识别期间: {result['intent']['fiscal_periods']}")
        print(f"使用 chunk 数: {result['context_count']}")
        print(f"来源: {result['sources'][:3]}")
        print(f"\n答案摘要（前500字）：")
        print(result["answer"][:500])

    print(f"\n{'='*60}")
    print("✅ 所有测试完成")


if __name__ == "__main__":
    run_tests()
```

---

## 执行顺序清单

```
【环境与依赖准备】
□ Step 0a: conda env list | grep tesla
□ Step 0b: conda run -n tesla pip install rank-bm25 anthropic
□ Step 0c: 将新包版本号追加到 requirements.txt

【创建文件】
□ Step 1:  追加检索配置到 core/config.py
□ Step 2:  创建 core/retrieval/__init__.py（空文件）
□ Step 3:  创建 core/retrieval/vector_store.py
□ Step 4:  创建 core/retrieval/bm25_retriever.py
□ Step 5:  创建 core/retrieval/hybrid_retriever.py
□ Step 6:  创建 core/retrieval/context_expander.py
□ Step 7:  创建 core/qa/__init__.py（空文件）
□ Step 8:  创建 core/qa/query_analyzer.py
□ Step 9:  创建 core/qa/answer_generator.py
□ Step 10: 创建 core/qa/pipeline.py
□ Step 11: 创建 tests/test_retrieval.py

【关键适配步骤（必须手动确认）】
□ Step 12: 在 tests/test_retrieval.py 的 get_embed_fn() 中
           填入与第一阶段入库时完全相同的 embedding 模型和调用方式
□ Step 13: 确认 core/config.py 中的
           chroma_collection_name 与第一阶段入库时使用的集合名一致
□ Step 14: 设置环境变量 ANTHROPIC_API_KEY

【验证运行】
□ Step 15: conda run -n tesla python tests/test_retrieval.py
□ Step 16: 如遇 ImportError → 立即安装并更新 requirements.txt
□ Step 17: 验证4种问题类型的答案质量
```

---

## 关键注意事项

1. **embedding 函数必须与入库时完全一致**：如果入库用 `BAAI/bge-large-en-v1.5`，检索也必须用同一个，否则向量空间不匹配，检索结果毫无意义
2. **BM25 索引首次构建较慢**：全量数据会加载到内存，构建后会缓存到 `db_data/bm25_index.pkl`，第二次启动秒级加载
3. **ChromaDB where 的 null 值处理**：第一阶段入库时 `quarter` 的 null 被转为 `-1`，过滤年报时应用 `{"quarter": {"$eq": -1}}`，而非 `null`
4. **ANTHROPIC_API_KEY 环境变量**：在运行前设置 `set ANTHROPIC_API_KEY=sk-...`（Windows）或 `export ANTHROPIC_API_KEY=sk-...`（Linux/Mac）
5. **所有命令必须在 tesla 环境中运行**：`conda run -n tesla python ...`
```
