# core/embedder/factory.py
"""
嵌入模型工厂函数。
根据配置字典选择并实例化对应的嵌入实现。
"""
import logging
from typing import Optional

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


def get_embedder(config: Optional[dict] = None) -> BaseEmbedder:
    """
    根据配置返回嵌入器实例。

    Args:
        config: 嵌入配置字典，支持以下字段：
            - provider (str): "local" 或 "api"，默认 "local"
            - local_model (str): 本地模型名称/路径
            - device (str|None): 设备名，None 表示自动检测（GPU 优先）
            - normalize_embeddings (bool): 是否 L2 归一化，默认 True
            - api_model (str): API 嵌入模型名称
            - api_key (str|None): API 密钥（可选，可由环境变量提供）
            - api_key_env (str): API Key 环境变量名，默认 "OPENAI_API_KEY"
            - api_base_url (str): OpenAI 兼容接口地址

    Returns:
        BaseEmbedder 子类实例

    Raises:
        ValueError: 不支持的 provider 类型
    """
    if config is None:
        config = {}

    provider = config.get("provider", "local").lower()
    logger.info(f"[EmbedderFactory] 使用嵌入提供者: {provider}")

    if provider == "local":
        from .local_embedder import LocalEmbedder
        return LocalEmbedder(
            model_name_or_path=config.get("local_model", "D:\\OneDrive\\Desktop\\大模型应用开发学习\\model\\BAAI\\bge-m3"),
            device=config.get("device", None),          # None → 自动检测（GPU 优先）
            normalize_embeddings=config.get("normalize_embeddings", True),
            show_progress_bar=config.get("show_progress_bar", False),
            use_fp16=config.get("use_fp16", None),      # None → CUDA 自动开启 fp16
        )

    elif provider == "api":
        from .api_embedder import ApiEmbedder
        return ApiEmbedder(
            model=config.get("api_model", "text-embedding-3-small"),
            api_key=config.get("api_key", None),
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=config.get("api_base_url", "https://api.openai.com/v1"),
            max_retries=config.get("max_retries", 3),
        )

    else:
        raise ValueError(
            f"不支持的嵌入提供者: '{provider}'。"
            f"请使用 'local' 或 'api'。"
        )
