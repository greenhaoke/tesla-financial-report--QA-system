# core/qa/pipeline.py
"""
完整问答流程串联：
QueryAnalyzer → HybridRetriever → ContextExpander → LLMReranker → AnswerGenerator
"""
import logging
from typing import List, Dict, Any, Optional

from core.qa.query_analyzer import QueryAnalyzer, QueryIntent
from core.retrieval.vector_store import VectorStore
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.context_expander import ContextExpander
from core.qa.answer_generator import AnswerGenerator
from core.retrieval.reranker import LLMReranker
from core.config import RETRIEVAL_CONFIG, RERANKER_CONFIG, EMBEDDING_CONFIG

logger = logging.getLogger(__name__)


class TeslaQAPipeline:
    """
    特斯拉财报智能问答系统主入口。

    使用方式：
        pipeline = TeslaQAPipeline()
        result = pipeline.ask("2022年Q3的汽车毛利率是多少？")
        print(result["answer"])

    也可以传入已实例化的 embedder 以复用模型（避免重复加载大模型）：
        from core.embedder.factory import get_embedder
        embedder = get_embedder(EMBEDDING_CONFIG)
        pipeline = TeslaQAPipeline(embedder=embedder)
    """

    def __init__(self, embedder=None):
        """
        Args:
            embedder: 可选，BaseEmbedder 实例。
                      若为 None，则根据 EMBEDDING_CONFIG 自动加载。
                      传入已有实例可避免重复加载大模型，节省显存和时间。
        """
        logger.info("[TeslaQAPipeline] 初始化...")

        # 加载 embedder（与入库时使用的必须相同！）
        if embedder is None:
            from core.embedder.factory import get_embedder
            logger.info("[TeslaQAPipeline] 加载 Embedder（首次加载较慢，大模型需要数十秒）...")
            embedder = get_embedder(EMBEDDING_CONFIG)
        self._embedder = embedder

        # 初始化检索组件
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever(vector_store=self.vector_store)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25,
            embedder=self._embedder,
        )
        self.expander = ContextExpander(vector_store=self.vector_store)

        # 初始化 LLM 组件
        self.analyzer = QueryAnalyzer()
        self.generator = AnswerGenerator()

        # 初始化重排序器（API 调用方式）
        if RERANKER_CONFIG.get("enabled", True):
            self.reranker = LLMReranker()
            logger.info("[TeslaQAPipeline] LLMReranker 已启用")
        else:
            self.reranker = None
            logger.info("[TeslaQAPipeline] LLMReranker 已禁用")

        logger.info("[TeslaQAPipeline] 初始化完成")

    def ask(
        self,
        question: str,
        verbose: bool = False,
        report_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        主问答接口。

        Args:
            question: 自然语言问题
            verbose: 为 True 时在返回中包含 retrieved_chunks 列表（调试用）
            report_type: 可选，显式指定报告类型（如 "10-K", "10-Q"），将强制添加到检索过滤条件中


        Returns:
            {
                "question": str,
                "answer": str,
                "sources": list,
                "intent": dict,           # 解析出的查询意图
                "retrieved_chunks": list, # 仅 verbose=True 时有数据
                "context_count": int,
                "usage": dict,            # LLM token 用量
            }
        """
        logger.info(f"[TeslaQAPipeline] 收到问题: {question}")

        # ── Step 1: 查询意图分析 ─────────────────────────────────
        intent = self.analyzer.analyze(question)

        if verbose:
            logger.info(f"[TeslaQAPipeline] 意图: {intent.__dict__}")

        # ── Step 1.5: 强制追加用户指定的报告类型过滤条件 ─────────
        if report_type and report_type in ["10-K", "10-Q"]:
            report_filter = {"report_type": {"$eq": report_type}}
            if not intent.metadata_filter:
                intent.metadata_filter = report_filter
            else:
                if "$and" in intent.metadata_filter:
                    intent.metadata_filter["$and"].append(report_filter)
                elif "$or" in intent.metadata_filter:
                    intent.metadata_filter = {"$and": [intent.metadata_filter, report_filter]}
                else:
                    # 单一条件
                    intent.metadata_filter = {"$and": [intent.metadata_filter, report_filter]}
            logger.info(f"[TeslaQAPipeline] 已强制添加 report_type 过滤: {intent.metadata_filter}")

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
            logger.info(f"[TeslaQAPipeline] 初始检索: {len(chunks)} 个 chunk")

        # ── Step 3: 上下文扩展（Small2Big）──────────────────────
        if intent.fiscal_periods and len(intent.fiscal_periods) > 1:
            chunks_by_period = {p: [] for p in intent.fiscal_periods}
            for c in chunks:
                p = c["metadata"].get("fiscal_period")
                if p in chunks_by_period:
                    chunks_by_period[p].append(c)
            expanded_chunks = self.expander.expand_by_period(chunks_by_period)
        else:
            expanded_chunks = self.expander.expand(chunks)

        if verbose:
            logger.info(f"[TeslaQAPipeline] 扩展后: {len(expanded_chunks)} 个 chunk")

        # ── Step 4: LLM 重排序（API 调用，精选最相关 chunk）──────
        if self.reranker is not None:
            final_chunks = self.reranker.rerank(
                query=question,
                chunks=expanded_chunks,
                top_k=RERANKER_CONFIG["top_k"],
                fiscal_periods=intent.fiscal_periods or None,  # 非空时触发期间均衡模式
            )
            if verbose:
                logger.info(f"[TeslaQAPipeline] 重排后: {len(final_chunks)} 个 chunk")

        else:
            # 未启用重排序时直接截断
            final_chunks = expanded_chunks[: RETRIEVAL_CONFIG["max_expanded_chunks"]]

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
                "needs_table": intent.needs_table,
                "needs_text": intent.needs_text,
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
        n_per_period = max(
            2,
            RETRIEVAL_CONFIG["final_top_k"] // max(len(intent.fiscal_periods), 1),
        )
        results_by_period = self.retriever.retrieve_multi_period(
            query=question,
            fiscal_periods=intent.fiscal_periods,
            n_per_period=n_per_period,
        )
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
