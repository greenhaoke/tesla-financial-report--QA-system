# core/embedder/api_embedder.py
"""
API 嵌入模型实现（OpenAI 兼容接口）。
支持 OpenAI、DeepSeek、通义千问等兼容 OpenAI API 的服务。
"""
import logging
import os
from typing import List, Optional

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class ApiEmbedder(BaseEmbedder):
    """
    通过 OpenAI 兼容接口进行嵌入的实现。

    支持的服务（需在 base_url 指定）：
    - OpenAI: https://api.openai.com/v1
    - DeepSeek: https://api.deepseek.com/v1
    - 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1
    - 本地 Ollama: http://localhost:11434/v1
    - 其他 OpenAI 兼容服务
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 3,
    ):
        """
        Args:
            model: 嵌入模型名称，如 "text-embedding-3-small"
            api_key: API 密钥（优先使用，若为 None 则从环境变量读取）
            api_key_env: API 密钥所在的环境变量名（默认 OPENAI_API_KEY）
            base_url: OpenAI 兼容接口地址
            max_retries: 请求失败时的最大重试次数
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai 未安装。\n"
                "请运行：pip install openai>=1.0.0"
            )

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise ValueError(
                f"API Key 未配置。请设置环境变量 {api_key_env}，"
                f"或在构造时通过 api_key 参数传入。"
            )

        self._model = model
        self._base_url = base_url
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        logger.info(
            f"[ApiEmbedder] 初始化完成 | 模型: {model} | "
            f"接口: {base_url}"
        )

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本列表（一次 API 调用）。

        Args:
            texts: 待嵌入文本（空列表直接返回 []）

        Returns:
            嵌入向量列表，顺序与输入一致
        """
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        # API 返回的 data 按 index 排序，保证顺序正确
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
