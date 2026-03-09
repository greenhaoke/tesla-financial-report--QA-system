# core/config.py
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
DB_DATA_DIR = ROOT_DIR / "db_data" / "chroma"

# ── 分块参数 ──────────────────────────────────────────────
CHUNK_CONFIG = {
    # 语义分块：段落合并的最大 token 数（超过则另起一块）
    "max_tokens_per_chunk": 512,
    # 段落合并的最小 token 数（太短的段落会向下合并）
    "min_tokens_to_merge": 50,
    # 块与块之间的重叠 token 数（保证上下文连续性）
    "overlap_tokens": 64,
    # 表格始终作为独立块，不与文本合并
    "table_as_standalone": True,
    # 章节标题识别的最大字符数（超过则不视为标题）
    "heading_max_chars": 120,
}

# ── 财报文档元数据 ─────────────────────────────────────────
# 文件名命名规范：tesla_{report_type}_{year}[_q{quarter}].pdf
# 例：tesla_10k_2022.pdf / tesla_10q_2023_q2.pdf

# ── Embedding 配置 ────────────────────────────────────────
EMBEDDING_CONFIG = {
    # 嵌入提供者："local"（本地模型）或 "api"（OpenAI 兼容接口）
    "provider": "local",
    # ── 本地模型配置 ──
    # HuggingFace 模型 ID 或本地路径（英文场景推荐 bge-small-en-v1.5，多语言用 bge-m3）
    "local_model": "D:\\OneDrive\\Desktop\\大模型应用开发学习\\model\\BAAI\\bge-m3",
    # 设备：None = 自动检测（GPU 优先: CUDA → MPS → CPU），也可指定 "cuda" / "cpu"
    "device": None,
    # 是否 L2 归一化（余弦相似度场景建议开启）
    "normalize_embeddings": True,
    # ── API 配置（provider="api" 时有效）──
    "api_base_url": "https://api.openai.com/v1",  # OpenAI 兼容接口地址
    "api_model": "text-embedding-3-small",         # API 嵌入模型名称
    "api_key_env": "OPENAI_API_KEY",              # API Key 所在环境变量
    # 是否使用 float16 半精度推理（CUDA 下减少约 50% 显存，解决 bge-m3 在 8GB 显存的 OOM）
    # None = CUDA 自动开启，CPU/MPS 自动关闭；True/False 可手动覆盖
    "use_fp16": None,
    # ── 批处理 ──
    "batch_size": 16,                              # bge-m3 在 8GB 显存的安全批量大小
}

# ChromaDB Collection 名称
CHROMA_COLLECTION = "tesla_financials"

# ── LLM 配置（OpenAI 兼容接口）────────────────────────────────
LLM_CONFIG = {
    # 接口类型：统一使用 openai 兼容接口
    "provider": "openai_compatible",
    # 主力模型（复杂推理/答案生成）
    "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    # 轻量模型（意图分析等简单任务）
    "fast_model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    # API Key 来源环境变量
    "api_key_env": "API_KEY",
    # OpenAI 兼容接口地址（可替换为 deepseek、通义千问等）
    "api_base_url": "https://api.siliconflow.cn/v1",
    "max_tokens": 4096,
    "temperature": 0.1,                # 财务问答要求精确，低温度
}

# ── 检索配置 ───────────────────────────────────────────────────
RETRIEVAL_CONFIG = {
    # ChromaDB collection 名称（必须与入库时一致）
    "chroma_collection_name": CHROMA_COLLECTION,

    # 混合检索权重（vector_weight + bm25_weight 不必须等于 1.0，RRF 自动归一化）
    "vector_weight": 0.6,
    "bm25_weight": 0.4,

    # 初始检索数量（Hybrid 合并前每路各取多少）
    "vector_top_k": 20,
    "bm25_top_k": 20,

    # 重排后保留的最终 chunk 数量
    "final_top_k": 8,

    # Small2Big 扩展配置
    "expand_by_section": True,         # 按 section_title 扩展同节
    "expand_neighbor_n": 1,            # 同时取相邻 N 个 chunk
    "max_expanded_chunks": 60,         # 扩展后最多保留的 chunk 数（需大于初始检索总量）

    # 表格专项检索
    "table_search_enabled": True,
    "table_top_k": 5,                  # 额外检索的表格数量

    # 数字意图关键词（触发表格优先检索）
    "numeric_intent_keywords": [
        "多少", "增长", "下降", "环比", "同比", "毛利率", "营收",
        "利润", "现金流", "revenue", "margin", "growth", "increase",
        "decrease", "quarter-over-quarter", "year-over-year", "QoQ", "YoY",
        "free cash flow", "FCF", "EPS", "gross profit",
    ],
}

# ── BM25 索引缓存路径 ──────────────────────────────────────────
BM25_INDEX_PATH = ROOT_DIR / "db_data" / "bm25_index.pkl"

# ── 重排序配置（API 调用）──────────────────────────────────────
RERANKER_CONFIG = {
    # 是否启用重排序（关闭时跳过该步骤，直接用 RRF 排序结果）
    "enabled": True,
    # 重排序后保留的 chunk 数量（传给 LLM 生成答案的最终数量）
    "top_k": 10,
    # 每次送入 LLM 做重排的最大 chunk 数（防止超出上下文长度）
    "batch_size": 30,
    # 每个 chunk 用于重排的最大内容截断字符数（节省 Token）
    "content_preview_chars": 800,

    # ── 多财务期间均衡重排配置（fiscal_periods 非空时生效）────────
    # 每个财务期间最终保留的 chunk 数量（如 3 个期间 × 6 = 18 chunks 传给 LLM）
    "top_k_per_period": 6,
    # 送入 LLM 做重排前，每个期间最多保留的候选 chunk 数（从扩展后分桶限制）
    "candidate_per_period": 25,
}
