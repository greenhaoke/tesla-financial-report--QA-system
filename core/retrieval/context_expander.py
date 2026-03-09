# core/retrieval/context_expander.py
"""
上下文扩展：Small2Big 策略实现。

核心思路：
- 向量检索找到的是精准的小块（small）
- 但理解需要完整章节上下文（big）
- 通过 section_title 和 chunk_index 扩展找到"大块"
"""
import logging
from typing import List, Dict, Any, Optional, Set

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
                    fiscal_period=meta.get("fiscal_period", ""),
                    section_title=meta["section_title"],
                )
                for sc in section_chunks:
                    if sc["id"] not in expanded_ids:
                        expanded_ids.add(sc["id"])
                        expanded_chunks.append(sc)

            # 策略2：相邻 chunk 扩展
            n = self.cfg["expand_neighbor_n"]
            if n > 0 and meta.get("fiscal_period") and meta.get("chunk_index") is not None:
                neighbor_chunks = self._get_neighbors(
                    fiscal_period=meta["fiscal_period"],
                    chunk_index=int(meta["chunk_index"]),
                    n=n,
                )
                for nc in neighbor_chunks:
                    if nc["id"] not in expanded_ids:
                        expanded_ids.add(nc["id"])
                        expanded_chunks.append(nc)

        # 限制总数，避免上下文过长（原始 chunk 优先保留，扩展块按剩余配额截取）
        max_chunks = self.cfg["max_expanded_chunks"]
        if len(expanded_chunks) > max_chunks:
            original_ids = {c["id"] for c in chunks}
            originals = [c for c in expanded_chunks if c["id"] in original_ids]
            extras = [c for c in expanded_chunks if c["id"] not in original_ids]
            # 原始块超过上限时直接截取原始块；否则用剩余配额填充扩展块
            if len(originals) >= max_chunks:
                expanded_chunks = originals[:max_chunks]
            else:
                quota = max_chunks - len(originals)
                expanded_chunks = originals + extras[:quota]

        # 按来源文档和页码排序，保证上下文连贯
        expanded_chunks.sort(key=lambda c: (
            c["metadata"].get("fiscal_period", ""),
            c["metadata"].get("page_start", 0),
            c["metadata"].get("chunk_index", 0),
        ))

        logger.info(
            f"[ContextExpander] 上下文扩展: {len(chunks)} → {len(expanded_chunks)} 个 chunk"
        )
        return expanded_chunks

    def expand_by_period(
        self,
        chunks_by_period: Dict[str, List[Dict[str, Any]]],
        max_per_period: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        按财务期间分别扩展，各期间独立控制 chunk 配额后合并。

        解决全局 expand() 中某期间 section 过大导致其他期间被挤出的问题。

        Args:
            chunks_by_period: {period: [seed_chunks]} 字典，由 retrieve_multi_period() 返回
            max_per_period: 每个期间扩展后最多保留的 chunk 数。
                            默认为 max_expanded_chunks // 期间数（至少 10）

        Returns:
            各期间扩展结果合并后的 chunk 列表（按 period + page + chunk_index 排序）
        """
        periods = [p for p, chunks in chunks_by_period.items() if chunks]
        if not periods:
            return []

        # 计算每期间配额
        if max_per_period is None:
            total_cap = self.cfg["max_expanded_chunks"]
            max_per_period = max(10, total_cap // len(periods))

        logger.info(
            f"[ContextExpander] 按期间分别扩展: periods={periods}, "
            f"max_per_period={max_per_period}"
        )

        all_expanded: List[Dict] = []
        for period in periods:
            seed_chunks = chunks_by_period[period]
            if not seed_chunks:
                logger.warning(f"[ContextExpander] 期间 [{period}] 无种子 chunk，跳过")
                continue

            # 临时覆盖 max_expanded_chunks 以限制单期间配额
            original_max = self.cfg["max_expanded_chunks"]
            self.cfg["max_expanded_chunks"] = max_per_period
            try:
                expanded = self.expand(seed_chunks)
            finally:
                self.cfg["max_expanded_chunks"] = original_max

            logger.info(
                f"[ContextExpander] [{period}] 扩展: {len(seed_chunks)} → {len(expanded)} 个 chunk"
            )
            all_expanded.extend(expanded)

        # 全局去重（跨期间 chunk_id 不会重复，但防御性保留）
        seen: Set[str] = set()
        result: List[Dict] = []
        for c in all_expanded:
            if c["id"] not in seen:
                seen.add(c["id"])
                result.append(c)

        result.sort(key=lambda c: (
            c["metadata"].get("fiscal_period", ""),
            c["metadata"].get("page_start", 0),
            c["metadata"].get("chunk_index", 0),
        ))

        summary = []
        for p in periods:
            orig_len = len(chunks_by_period.get(p, []))
            new_len = sum(1 for c in result if c["metadata"].get("fiscal_period") == p)
            summary.append(f"{p}:{orig_len}→{new_len}")
        summary_str = ", ".join(summary)

        logger.info(
            f"[ContextExpander] 期间分别扩展完成: 总计 {len(result)} 个 chunk ({summary_str})"
        )
        return result


    def _get_same_section(
        self, fiscal_period: str, section_title: str
    ) -> List[Dict]:
        """获取同一财报同一章节的所有 chunk。"""
        year, quarter = self._parse_fiscal_period(fiscal_period)
        where = self._build_period_where(year, quarter)

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

        target_indices = (
            list(range(max(0, chunk_index - n), chunk_index))
            + list(range(chunk_index + 1, chunk_index + n + 1))
        )

        results = []
        for idx in target_indices:
            idx_where = {"$and": where.get("$and", [where])[:] }
            if "$and" in idx_where:
                idx_where["$and"] = list(idx_where["$and"]) + [{"chunk_index": {"$eq": idx}}]
            else:
                idx_where = {"$and": [idx_where, {"chunk_index": {"$eq": idx}}]}
            chunks = self.vs.get_by_metadata_filter(where=idx_where, limit=1)
            results.extend(chunks)
        return results

    def _parse_fiscal_period(self, fiscal_period: str):
        """解析 fiscal_period，返回 (year, quarter)。"""
        if not fiscal_period:
            return None, None
        if fiscal_period.upper().startswith("FY"):
            return int(fiscal_period[2:]), None
        else:
            year = int(fiscal_period[:4])
            quarter = int(fiscal_period[5]) if len(fiscal_period) > 5 else None
            return year, quarter

    def _build_period_where(self, year, quarter) -> Dict:
        """根据年份和季度构建 ChromaDB where 条件。"""
        if year is None:
            return {}
        if quarter is None:
            # 年报：quarter 入库时为 -1
            return {"$and": [
                {"year": {"$eq": year}},
                {"quarter": {"$eq": -1}},
            ]}
        else:
            return {"$and": [
                {"year": {"$eq": year}},
                {"quarter": {"$eq": quarter}},
            ]}
