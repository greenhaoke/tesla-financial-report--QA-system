# core/embedder/local_embedder.py
"""
本地嵌入模型实现（基于 sentence-transformers）。
GPU 优先：自动检测 CUDA，回退至 CPU。
"""
import logging
from typing import List, Optional

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


def _detect_device(preferred: Optional[str] = None) -> str:
    """
    自动选择设备。

    Args:
        preferred: 用户指定的设备（"cuda" / "cpu" / "mps" / None）
                   None 表示自动检测。

    Returns:
        设备字符串，如 "cuda", "cuda:0", "mps", "cpu"
    """
    if preferred is not None:
        return preferred

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"[LocalEmbedder] GPU 可用，使用 {device} ({gpu_name})")
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[LocalEmbedder] Apple MPS 可用，使用 mps")
            return "mps"
        else:
            logger.info("[LocalEmbedder] GPU 不可用，回退到 CPU")
            return "cpu"
    except ImportError:
        logger.warning("[LocalEmbedder] torch 未安装，使用 CPU")
        return "cpu"


class LocalEmbedder(BaseEmbedder):
    """
    使用 sentence-transformers 加载本地或 HuggingFace 模型的嵌入实现。

    GPU 优先策略：
    - 自动检测 CUDA → MPS → CPU
    - 可通过 device 参数手动指定
    - CUDA 设备上默认使用 float16（半精度），显存占用减半，解决 8GB 显存 OOM 问题
    """

    def __init__(
        self,
        model_name_or_path: str = "D:\\OneDrive\\Desktop\\大模型应用开发学习\\model\\BAAI\\bge-m3",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        use_fp16: Optional[bool] = None,
    ):
        """
        Args:
            model_name_or_path: HuggingFace 模型 ID 或本地模型路径
            device: 设备（None 表示自动检测，优先 CUDA）
            normalize_embeddings: 是否 L2 归一化（余弦相似度场景建议开启）
            show_progress_bar: 批量嵌入时是否显示进度条
            use_fp16: 是否使用 float16 半精度推理（默认 CUDA 设备自动开启，CPU/MPS 关闭）
                      bge-m3 等大模型在 8GB 显存下必须开启才能避免 OOM。
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装。\n"
                "请运行：pip install sentence-transformers>=2.2.0"
            )

        self._device = _detect_device(device)
        self._normalize = normalize_embeddings
        self._show_progress = show_progress_bar
        self._model_name = model_name_or_path

        logger.info(f"[LocalEmbedder] 加载模型 '{model_name_or_path}'，设备: {self._device}")
        self._model = SentenceTransformer(model_name_or_path, device=self._device)

        # 半精度推理：CUDA 设备默认开启（减少约 50% 显存），CPU/MPS 默认关闭
        if use_fp16 is None:
            use_fp16 = self._device.startswith("cuda")
        self._use_fp16 = use_fp16

        if self._use_fp16:
            try:
                import torch
                self._model = self._model.half()  # float32 → float16，显存减半
                logger.info("[LocalEmbedder] 已启用 float16 半精度推理（显存减半，解决大模型 OOM）")
            except Exception as e:
                logger.warning(f"[LocalEmbedder] float16 转换失败，回退 float32: {e}")
                self._use_fp16 = False

        logger.info(
            f"[LocalEmbedder] 模型加载完成，维度: {self._model.get_sentence_embedding_dimension()}，"
            f"精度: {'float16' if self._use_fp16 else 'float32'}"
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本列表。

        Args:
            texts: 待嵌入文本（空列表直接返回 []）

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        try:
            import torch
            ctx = torch.no_grad()
        except ImportError:
            import contextlib
            ctx = contextlib.nullcontext()

        with ctx:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self._normalize,
                show_progress_bar=self._show_progress,
                convert_to_numpy=True,
            )

        # 推理完成后立即释放 CUDA 显存碎片，防止大模型（bge-m3 等）批次间 OOM
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # fp16 模型输出 float16 numpy 数组；ChromaDB 需要 float32，做一次类型转换
        # （内存中仅此一个中间数组，不会额外占用 VRAM）
        if self._use_fp16 and embeddings.dtype.name == "float16":
            embeddings = embeddings.astype("float32")

        return embeddings.tolist()

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
