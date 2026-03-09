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
    1. 保留金融术语短语（用下划线连接）
    2. 保留数字+单位（如 $1.2B, 23.5%）
    3. 小写化，移除无意义标点
    """
    text_lower = text.lower()

    # 用占位符保护金融短语
    placeholders: Dict[str, str] = {}
    for phrase in FINANCIAL_PHRASES:
        if phrase in text_lower:
            ph = f"__PHRASE_{len(placeholders)}__"
            placeholders[ph] = phrase.replace(" ", "_")
            text_lower = text_lower.replace(phrase, ph)

    # 保留数字相关 token
    tokens = re.findall(
        r'\$[\d,.]+[bm]?\b'     # 金额
        r'|\d+\.?\d*%'           # 百分比
        r'|q[1-4]\b'             # 季度
        r'|fy\d{4}\b'            # 财年
        r'|\d{4}(?:q[1-4])?\b'  # 年份/年季
        r'|[a-z_]+',             # 普通词和占位符
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
        """
        Args:
            vector_store: VectorStore 实例，首次构建索引时使用。
                          如果磁盘缓存存在则不需要此参数。
        """
        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._vector_store = vector_store

        if BM25_INDEX_PATH.exists():
            self._load_index()
        elif vector_store is not None:
            self.build_index(vector_store.get_all_for_bm25())
        else:
            logger.warning(
                "[BM25Retriever] 索引未加载，请先调用 build_index() 或传入 vector_store"
            )

    def build_index(self, chunks: List[Dict[str, Any]]):
        """从 chunk 列表构建 BM25 索引并缓存到磁盘。"""
        logger.info(f"[BM25Retriever] 构建索引，共 {len(chunks)} 条...")
        self.corpus = chunks
        self.tokenized_corpus = [tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
        logger.info("[BM25Retriever] 索引构建完成并已缓存")

    def search(
        self,
        query: str,
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索，支持 where 过滤（在 BM25 结果上做后过滤）。

        注意：BM25 不原生支持预过滤，通过后过滤实现。
        """
        if self.bm25 is None:
            logger.error("[BM25Retriever] 索引未初始化")
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 取 top n*5 留余量供过滤
        n_fetch = min(n_results * 5, len(self.corpus))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_fetch]

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
            pickle.dump(
                {"corpus": self.corpus, "tokenized": self.tokenized_corpus}, f
            )
        logger.info(f"[BM25Retriever] 索引已保存: {BM25_INDEX_PATH}")

    def _load_index(self):
        logger.info(f"[BM25Retriever] 加载索引: {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.tokenized_corpus = data["tokenized"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"[BM25Retriever] 索引加载完成: {len(self.corpus)} 条")
