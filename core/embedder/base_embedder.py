# core/embedder/base_embedder.py
"""
嵌入模型抽象基类。
所有嵌入实现（本地、API）均继承此类，统一接口。
"""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """嵌入模型抽象基类，定义统一的嵌入接口。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表，每个向量为 float 列表
        """
        ...

    def embed_one(self, text: str) -> List[float]:
        """
        嵌入单条文本的便利方法。

        Args:
            text: 单条文本

        Returns:
            嵌入向量（float 列表）
        """
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        """
        返回嵌入维度。
        子类可覆盖；默认通过嵌入空字符串探测维度。
        """
        return len(self.embed_one("dimension probe"))
