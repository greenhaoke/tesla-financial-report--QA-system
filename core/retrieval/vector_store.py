# core/retrieval/vector_store.py
"""
ChromaDB 连接封装，提供带元数据过滤的向量检索接口。
"""
import os
import logging
from typing import List, Optional, Dict, Any

# 禁用 ChromaDB 匿名遥测
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb import Collection

from core.config import DB_DATA_DIR, RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 向量存储封装。"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_DATA_DIR))
        collection_name = RETRIEVAL_CONFIG["chroma_collection_name"]
        self.collection: Collection = self.client.get_collection(name=collection_name)
        logger.info(
            f"[VectorStore] 已连接 collection '{collection_name}'，"
            f"共 {self.collection.count()} 条记录"
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
            where: ChromaDB 元数据过滤条件，例如：
                   {"year": {"$eq": 2022}}
                   {"$and": [{"year": {"$eq": 2022}}, {"quarter": {"$eq": 3}}]}
            where_document: 文档内容过滤

        Returns:
            标准化的 chunk 字典列表，每个包含 id/content/metadata/score/source
        """
        total = self.collection.count()
        if total == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, total),
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
                # ChromaDB cosine distance ∈ [0,2]，转换为 [0,1] 相似度
                "score": 1.0 - raw["distances"][0][i] / 2.0,
                "source": "vector",
            })
        return results

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """按 chunk_id 批量获取，用于 Small2Big 扩展。"""
        if not ids:
            return []
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
                "score": 1.0,
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
            elif "$or" in actual_where:
                actual_where = {"$and": [actual_where, type_filter]}
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
        logger.info(f"[VectorStore] 加载全量数据用于 BM25: {len(results)} 条")
        return results
