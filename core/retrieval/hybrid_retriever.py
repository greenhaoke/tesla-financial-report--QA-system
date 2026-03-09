# core/retrieval/hybrid_retriever.py
"""
混合检索：向量检索（语义）+ BM25（关键词）→ RRF 融合排序

RRF（Reciprocal Rank Fusion）：一种简单有效的多路检索融合算法，
不需要对两路分数做归一化，只依赖排名位置。
"""
import logging
from typing import List, Dict, Any, Optional

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

    fused = []
    for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(doc_data[doc_id])
        item["rrf_score"] = score
        item["score"] = score
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

    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25Retriever, embedder):
        """
        Args:
            vector_store: VectorStore 实例
            bm25_retriever: BM25Retriever 实例
            embedder: BaseEmbedder 实例（与入库时使用的相同 embedding 模型）
        """
        self.vs = vector_store
        self.bm25 = bm25_retriever
        self.embedder = embedder
        self.cfg = RETRIEVAL_CONFIG

    def _embed(self, text: str) -> List[float]:
        """使用 embedder 将文本转为向量。"""
        return self.embedder.embed_one(text)

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        where: Optional[Dict] = None,
        force_table_search: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        主检索入口。

        Args:
            query: 用户查询文本
            n_results: 最终返回数量（默认用 config 中的 final_top_k * 3，供后续扩展）
            where: 元数据过滤条件（由 QueryAnalyzer 生成）
            force_table_search: 强制触发表格专项检索
        """
        n_results = n_results or self.cfg["final_top_k"]

        # 1. 生成查询向量
        query_vec = self._embed(query)

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
            table_results = self.vs.get_by_metadata_filter(
                where=dict(where),
                chunk_type="table",
                limit=self.cfg["table_top_k"],
            )
            logger.info(f"[HybridRetriever] 表格专项检索: 找到 {len(table_results)} 张表格")

        # 5. RRF 融合
        result_lists = [vec_results, bm25_results]
        weights = [self.cfg["vector_weight"], self.cfg["bm25_weight"]]
        if table_results:
            result_lists.append(table_results)
            weights.append(0.3)

        fused = rrf_fusion(result_lists, weights)
        final = fused[: n_results * 3]

        logger.info(
            f"[HybridRetriever] 混合检索完成: 向量{len(vec_results)}"
            f" + BM25{len(bm25_results)} + 表格{len(table_results)}"
            f" → 融合后{len(fused)}，取前{len(final)}"
        )
        # ── DEBUG: 打印首次检索返回的 chunk ID ──────────────────────
        logger.debug("[HybridRetriever] 首次检索 chunk IDs (按 RRF 排序):")
        for rank, c in enumerate(final, 1):
            logger.debug(
                f"  [{rank:>3}] id={c['id']}  rrf_score={c.get('rrf_score', 0):.5f}"
                f"  period={c['metadata'].get('fiscal_period','?')}"
                f"  section={c['metadata'].get('section_title','')[:30]}"
            )
        # ─────────────────────────────────────────────────────────────
        # 返回更多，供后续扩展和重排使用
        return final

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
        results_by_period: Dict[str, List[Dict]] = {}
        query_vec = self._embed(query)

        for period in fiscal_periods:
            # 根据 period 格式判断是年报还是季报
            if period.upper().startswith("FY"):
                year = int(period[2:])
                where: Dict = {"$and": [
                    {"year": {"$eq": year}},
                    {"quarter": {"$eq": -1}},  # 年报 quarter 入库时转换为 -1
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
            logger.info(f"[HybridRetriever] [{period}] 检索到 {len(fused)} 条结果")

        return results_by_period

    def _has_numeric_intent(self, query: str) -> bool:
        """判断问题是否包含数字类意图（需要表格支持）。"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.cfg["numeric_intent_keywords"])
