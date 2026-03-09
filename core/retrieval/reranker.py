# core/retrieval/reranker.py
"""
重排序器：使用 LLM API（OpenAI 兼容接口）对候选 chunk 进行相关性打分。

策略：
- 将候选 chunk（截断预览）+ 问题一次性发给 LLM
- 要求 LLM 返回按相关性排序的编号列表（JSON 格式）
- 依据排序结果重排 chunk，输出 top-K

多财务期间均衡模式（fiscal_periods 非空时）：
- 按 metadata.fiscal_period 分桶，每组独立 LLM 重排
- 每组取 top_k_per_period 个，round-robin 交错合并
- 保证 FY2023/FY2024/FY2025 等各期间均有均等代表

优点：
- 只需 1 次 API 调用（batch 策略，大于 batch_size 时分批处理后合并）
- 不依赖本地模型，无显存开销
- 使用 fast_model 控制成本
"""
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI

from core.config import LLM_CONFIG, RERANKER_CONFIG

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = """你是一个专业的文档相关性判断助手。
你的任务是评估给定文档段落与用户问题的相关性，并按相关性从高到低排序。
只返回 JSON 格式的编号列表，不要任何其他文字。"""

RERANK_USER_PROMPT = """请根据以下用户问题，对候选文档段落按相关性从高到低排序。

用户问题：{question}

候选文档段落（编号从 1 开始）：
{candidates}

请返回按相关性排序的编号列表（最相关排第一）。
格式：{{"ranked_ids": [3, 1, 5, 2, 4]}}
只返回 JSON，不要有其他文字。"""


class LLMReranker:
    """
    基于 LLM API 的重排序器。

    使用 OpenAI 兼容接口（fast_model），一次 API 调用完成候选列表的相关性排序。

    支持两种模式：
    - 全局模式（fiscal_periods 为空）：对所有 chunk 做全局排序，取 top_k
    - 期间均衡模式（fiscal_periods 非空）：按财务期间分桶分别排序，各取 top_k_per_period，
      再 round-robin 交错合并，保证各报告均等代表
    """

    def __init__(self):
        api_key = os.environ.get(LLM_CONFIG["api_key_env"])
        if not api_key:
            raise ValueError(
                f"重排序器 API Key 未配置，请设置环境变量: {LLM_CONFIG['api_key_env']}"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=LLM_CONFIG["api_base_url"],
        )
        # 重排序用轻量模型（节省成本）
        self.model = LLM_CONFIG["fast_model"]
        self.cfg = RERANKER_CONFIG

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        fiscal_periods: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        对候选 chunk 列表重排序，返回最相关的结果。

        Args:
            query: 用户查询文本
            chunks: 候选 chunk 列表（已经过 RRF 初步融合排序）
            top_k: 全局模式下的返回数量，默认使用 RERANKER_CONFIG["top_k"]
            fiscal_periods: 财务期间列表（如 ['FY2023', 'FY2024', 'FY2025']）。
                非空时启用期间均衡模式，每个期间各取 top_k_per_period 个 chunk。

        Returns:
            重排后的 chunk 列表（已添加 rerank_score 字段）
        """
        if not self.cfg.get("enabled", True):
            logger.info("[LLMReranker] 重排序已禁用，跳过")
            return chunks[: top_k or self.cfg["top_k"]]

        if not chunks:
            return []

        # 有明确 fiscal_periods 时启用期间均衡模式
        if fiscal_periods:
            return self.rerank_by_period(query, chunks, fiscal_periods)

        # ── 全局模式（原有逻辑）────────────────────────────────────
        top_k = top_k or self.cfg["top_k"]
        batch_size = self.cfg["batch_size"]

        logger.info(
            f"[LLMReranker] 全局重排序: {len(chunks)} 个候选 chunk → top {top_k}"
        )

        # 超出 batch_size 时分批处理
        if len(chunks) <= batch_size:
            ranked = self._rerank_batch(query, chunks)
        else:
            # 分批：先各批内重排取半，再合并重排一次
            ranked = self._rerank_in_batches(query, chunks, batch_size)

        # 截取 top_k 并记录重排分数
        result = []
        for rank, chunk in enumerate(ranked[:top_k]):
            item = dict(chunk)
            item["rerank_score"] = 1.0 - rank / max(len(ranked), 1)
            item["score"] = item["rerank_score"]
            result.append(item)

        logger.info(f"[LLMReranker] 全局重排完成，最终保留 {len(result)} 个 chunk")
        # ── DEBUG: 打印最终保留的 chunk IDs 及 rerank_score ─────────
        logger.debug("[LLMReranker] 最终保留 chunk IDs (按 rerank_score 降序):")
        for item in result:
            logger.debug(
                f"  id={item['id']}  rerank_score={item['rerank_score']:.4f}"
                f"  period={item['metadata'].get('fiscal_period','?')}"
                f"  section={item['metadata'].get('section_title','')[:30]}"
            )
        # ─────────────────────────────────────────────────────────────
        return result

    def rerank_by_period(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        fiscal_periods: List[str],
    ) -> List[Dict[str, Any]]:
        """
        按财务期间均衡重排序。

        步骤：
        1. 将 chunks 按 metadata.fiscal_period 分桶
        2. 对每个期间的 chunk 候选（上限 candidate_per_period）独立 LLM 重排
        3. 每组取前 top_k_per_period 个，附加 rerank_score
        4. round-robin 交错合并，保证各期间均等代表

        Args:
            query: 用户查询文本
            chunks: 扩展后的全量候选 chunk
            fiscal_periods: 期望覆盖的期间列表，如 ['FY2023', 'FY2024', 'FY2025']

        Returns:
            均衡重排后的 chunk 列表
        """
        top_k_per_period = self.cfg.get("top_k_per_period", 6)
        candidate_per_period = self.cfg.get("candidate_per_period", 20)

        logger.info(
            f"[LLMReranker] 期间均衡重排: periods={fiscal_periods}, "
            f"top_k_per_period={top_k_per_period}, "
            f"总候选={len(chunks)} 个 chunk"
        )

        # Step 1: 按期间分桶（只保留意图期间的 chunk，其余忽略）
        buckets: Dict[str, List[Dict]] = {p: [] for p in fiscal_periods}
        uncategorized: List[Dict] = []

        for chunk in chunks:
            period = chunk["metadata"].get("fiscal_period", "")
            if period in buckets:
                buckets[period].append(chunk)
            else:
                uncategorized.append(chunk)

        for period, bucket in buckets.items():
            logger.info(f"[LLMReranker]   分桶 [{period}]: {len(bucket)} 个 chunk")
        if uncategorized:
            logger.info(
                f"[LLMReranker]   未归类（不在期间列表中）: {len(uncategorized)} 个"
            )

        # Step 2: 每桶独立重排，各取 top_k_per_period
        ranked_by_period: Dict[str, List[Dict]] = {}
        for period in fiscal_periods:
            bucket = buckets[period]
            if not bucket:
                logger.warning(f"[LLMReranker] 期间 [{period}] 无候选 chunk，跳过")
                ranked_by_period[period] = []
                continue

            # 限制候选数量（防止单桶过多塞满上下文）
            candidates = bucket[:candidate_per_period]
            logger.info(
                f"[LLMReranker] 对 [{period}] 的 {len(candidates)} 个候选进行重排..."
            )

            ranked = self._rerank_batch(query, candidates)

            # 取 top_k_per_period 并赋 rerank_score
            period_result = []
            for rank, chunk in enumerate(ranked[:top_k_per_period]):
                item = dict(chunk)
                item["rerank_score"] = 1.0 - rank / max(len(ranked), 1)
                item["score"] = item["rerank_score"]
                period_result.append(item)

            ranked_by_period[period] = period_result
            logger.info(
                f"[LLMReranker] [{period}] 重排完成，保留 {len(period_result)} 个 chunk"
            )
            logger.debug(f"[LLMReranker] [{period}] 保留 chunk IDs:")
            for item in period_result:
                logger.debug(
                    f"  id={item['id']}  rerank_score={item['rerank_score']:.4f}"
                    f"  section={item['metadata'].get('section_title','')[:30]}"
                )

        # Step 3: round-robin 交错合并
        result: List[Dict] = []
        max_len = max((len(v) for v in ranked_by_period.values()), default=0)
        for i in range(max_len):
            for period in fiscal_periods:
                lst = ranked_by_period.get(period, [])
                if i < len(lst):
                    result.append(lst[i])

        logger.info(
            f"[LLMReranker] 期间均衡重排完成，最终合并 {len(result)} 个 chunk "
            f"({', '.join(f'{p}:{len(ranked_by_period.get(p,[]))}' for p in fiscal_periods)})"
        )
        return result

    def _rerank_batch(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """单批重排序（chunks 数量 ≤ batch_size）。"""
        preview_chars = self.cfg["content_preview_chars"]

        # 构建候选段落文本
        candidates_text = ""
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            period = meta.get("fiscal_period", "?")
            section = meta.get("section_title", "")[:30]
            chunk_type = meta.get("chunk_type", "text")
            preview = chunk["content"][:preview_chars].replace("\n", " ")
            candidates_text += (
                f"\n[{i+1}] [{period} | {section} | {chunk_type}]\n{preview}\n"
            )

        prompt = RERANK_USER_PROMPT.format(
            question=query,
            candidates=candidates_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # 提取 JSON
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group()
            data = json.loads(raw)
            ranked_ids = data.get("ranked_ids", [])

            # ── DEBUG: 打印 LLM 返回的排序编号及对应 chunk ID ────────
            logger.debug(f"[LLMReranker] 本批输入 chunk IDs:")
            for i, c in enumerate(chunks, 1):
                logger.debug(f"  [{i:>2}] id={c['id']}")
            logger.debug(f"[LLMReranker] LLM 返回排序: {ranked_ids}")
            # ─────────────────────────────────────────────────────────

            # 按 LLM 给出的顺序重排（1-indexed → 0-indexed）
            used = set()
            ranked_chunks = []
            for rank_id in ranked_ids:
                idx = int(rank_id) - 1
                if 0 <= idx < len(chunks) and idx not in used:
                    ranked_chunks.append(chunks[idx])
                    used.add(idx)

            # 补充未被 LLM 列出的 chunk（保证完整性）
            for i, chunk in enumerate(chunks):
                if i not in used:
                    ranked_chunks.append(chunk)

            # ── DEBUG: 打印本批重排后的 chunk ID 顺序 ────────────────
            logger.debug(f"[LLMReranker] 本批重排后 chunk IDs (按相关性降序):")
            for rank, c in enumerate(ranked_chunks, 1):
                logger.debug(f"  [{rank:>2}] id={c['id']}")
            # ─────────────────────────────────────────────────────────
            return ranked_chunks

        except Exception as e:
            logger.warning(
                f"[LLMReranker] 重排序 API 调用失败: {e}，回退到原始顺序"
            )
            return chunks

    def _rerank_in_batches(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """
        分批重排：各批各取前半，再合并做一轮最终重排。
        """
        batches = [
            chunks[i: i + batch_size] for i in range(0, len(chunks), batch_size)
        ]
        half = max(1, batch_size // 2)
        candidates_for_final = []

        for b_idx, batch in enumerate(batches):
            logger.info(
                f"[LLMReranker] 批次 {b_idx+1}/{len(batches)} 重排 {len(batch)} 个 chunk"
            )
            ranked_batch = self._rerank_batch(query, batch)
            top_half = ranked_batch[:half]
            logger.debug(
                f"[LLMReranker] 批次 {b_idx+1} 取前 {len(top_half)} 个进入合并池:"
                f" {[c['id'] for c in top_half]}"
            )
            candidates_for_final.extend(top_half)

        # 合并后做最终重排
        logger.info(
            f"[LLMReranker] 合并 {len(candidates_for_final)} 个候选进行最终重排"
        )
        logger.debug(
            f"[LLMReranker] 最终重排输入 chunk IDs: {[c['id'] for c in candidates_for_final]}"
        )
        return self._rerank_batch(query, candidates_for_final)
