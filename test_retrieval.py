# test_retrieval.py
"""
检索与问答流程端到端测试脚本。

用法：
  # 仅测试检索（不调 LLM，不消耗 Token）
  conda run -n tesla python test_retrieval.py --mode retrieval

  # 完整 QA 测试（需要设置 OPENAI_API_KEY 或对应的 LLM 环境变量）
  conda run -n tesla python test_retrieval.py --mode qa

  # 测试单条问题
  conda run -n tesla python test_retrieval.py --mode qa --question "2023年特斯拉的总营收是多少？"
"""
import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# ── 项目路径 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 禁用 ChromaDB 遥测
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# 单独为 retrieval 模块开启 DEBUG，方便追踪 chunk ID 流转
logging.getLogger("core.retrieval").setLevel(logging.DEBUG)
logger = logging.getLogger("test_retrieval")

# ── 测试用例 ──────────────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "q": "比较2023年和2025年财报中关于“政府激励措施（Government Incentives）”对电动汽车需求影响的描述。特别是2025年财报中新提到的哪项法案对激励政策产生了重大改变，其具体影响是什么？",
        "type": "",
        "desc": "Cross-document Longitudinal Comparison + Semantic Change Detection",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "比较2023年、2024年和2025年“风险因素”中关于“高度依赖埃隆·马斯克”的描述变化，特别是他在其他公司或政府机构任职的情况。",
        "type": "",
        "desc": "Numerical Calculation + Cross-year Comparison",
        "expect_period": ["FY2023", "FY2024"],
    },
    {
        "q": "特斯拉在2024年和2025年分别进行了重组。请根据文件描述这两次重组发生的时间段、涉及的主要开支金额及其重组的目的。",
        "type": "",
        "desc": "Text + Table Association + Implicit Reasoning",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "对比2023、2024、2025三份财报中管理层对“下一年资本支出（Capital Expenditures）”的预测值。哪份文件首次大幅调高了未来支出的阈值？",
        "type": "",
        "desc": "Multi-indicator Correlation + Trend Judgment",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "描述“2018年CEO绩效奖励”相关的法律纠纷在2024年和2025年发生的关键转折点。根据2025年财报，该诉讼的最终法律地位是什么？",
        "type": "",
        "desc": "Implicit Calculation + Unit Conversion",
        "expect_period": ["FY2023", "FY2024"],
    },
    {
        "q": "比较2025年 Q2 和 Q3 财报中关于“One Big Beautiful Bill Act (OBBBA)”对特斯拉业务影响的描述。Q3 相比 Q2 在风险评估上增加了哪些具体的财务担忧？",
        "type": "",
        "desc": "Cross-document Longitudinal Comparison + Semantic Change Detection",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "在 2025 年前三个季度中，哪一个季度的“总汽车业务毛利率（Gross margin total automotive）”最低？请提取该季度 MD&A 部分中解释该利润率下滑的三个关键因素。",
        "type": "",
        "desc": "Numerical Calculation + Cross-year Comparison",
        "expect_period": ["FY2023", "FY2024"],
    },
    {
        "q": "根据 Q1 和 Q2 的 MD&A 部分，特斯拉在 2025 年上半年对于“汽车监管信用额度（Regulatory Credits）”收入变动的解释有何本质不同？",
        "type": "",
        "desc": "Text + Table Association + Implicit Reasoning",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "根据财报记录，特斯拉在 2025 年哪个月份首次推出了 Robotaxi 服务？该服务首发的城市是哪里？在 Q2 和 Q3 财报中，管理层对于扩展该业务所强调的“专用基础设施”涵盖哪些方面？",
        "type": "",
        "desc": "Multi-indicator Correlation + Trend Judgment",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "比较2025 年 Q1 和 Q2 的“能源发电与存储”业务。尽管 Q2 的该板块营收（27.89 亿）高于 Q1（27.30 亿），但其毛利率有何变化？MD&A 提到的导致这种成本结构优化的主要原材料因素是什么？",
        "type": "",
        "desc": "Implicit Calculation + Unit Conversion",
        "expect_period": ["FY2023", "FY2024"],
    },
    {
        "q": "Q1 财报提到因“同时关闭全球工厂”以切换新 Model Y 生产线导致产量损失。请从 Q1 资产负债表提取“成品库存（Finished goods）”数据，并结合 MD&A 说明 2025 年第一季度交付量下滑的具体数字。。",
        "type": "",
        "desc": "Numerical Calculation + Cross-year Comparison",
        "expect_period": ["FY2023", "FY2024"],
    },
    {
        "q": "根据2025年财报中的合并现金流量表，2025年“经营活动提供的净现金”较2024年减少了约1.76亿美元，请通过提取表中“净利润”和“递延所得税”的变化来解释这一变动的主要技术性原因。",
        "type": "",
        "desc": "Text + Table Association + Implicit Reasoning",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "2025 年前三个季度中，哪一个季度的研发费用（R&D）占总营收比例最高？该比例是多少？Q3 财报中提到研发费用增长除了 AI 项目外，还受什么支出的影响？",
        "type": "",
        "desc": "Multi-indicator Correlation + Trend Judgment",
        "expect_period": ["FY2023", "FY2024", "FY2025"],
    },
    {
        "q": "追踪 2025 年三个季度特斯拉持有的比特币数量及账面价值变动。哪份财报显示发生了“市场价损失（mark-to-market loss）”？该损失的具体金额是多少？",
        "type": "",
        "desc": "Implicit Calculation + Unit Conversion",
        "expect_period": ["FY2023", "FY2024"],
    },
]


def _print_separator(title: str = "", width: int = 65):
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print(f"\n{'=' * pad} {title} {'=' * pad}")
    else:
        print("=" * width)


# ── 检索模式测试 ───────────────────────────────────────────────
def run_retrieval_tests(questions: List[Dict] = None):
    """
    仅测试检索流程（向量 + BM25 + 混合），不调用 LLM。
    用于快速验证 ChromaDB 连接、BM25 索引构建、混合检索效果。
    """
    from core.embedder.factory import get_embedder
    from core.config import EMBEDDING_CONFIG
    from core.retrieval.vector_store import VectorStore
    from core.retrieval.bm25_retriever import BM25Retriever
    from core.retrieval.hybrid_retriever import HybridRetriever
    from core.retrieval.context_expander import ContextExpander

    _print_separator("初始化检索组件")

    logger.info("加载 Embedder（本地模型首次加载较慢）...")
    embedder = get_embedder(EMBEDDING_CONFIG)
    logger.info(f"Embedder 加载完成，嵌入维度: {embedder.embedding_dim}")

    vector_store = VectorStore()
    logger.info(f"ChromaDB 连接成功，记录总数: {vector_store.collection.count()}")

    bm25 = BM25Retriever(vector_store=vector_store)
    retriever = HybridRetriever(vector_store, bm25, embedder)
    expander = ContextExpander(vector_store)

    questions = questions or TEST_QUESTIONS

    passed = 0
    failed = 0

    for i, test in enumerate(questions):
        _print_separator(f"测试 {i+1}/{len(questions)}: {test.get('desc', test['type'])}")
        print(f"问题: {test['q']}")
        print("-" * 65)

        try:
            # 向量检索
            query_vec = embedder.embed_one(test["q"])
            vec_results = vector_store.query(query_embedding=query_vec, n_results=5)
            print(f"[向量检索] 找到 {len(vec_results)} 条")
            for r in vec_results[:3]:
                meta = r["metadata"]
                print(
                    f"  · [{r['score']:.3f}] {meta.get('fiscal_period', '?')} | "
                    f"{meta.get('section_title', '')[:40]} | "
                    f"type={meta.get('chunk_type', '?')}"
                )

            # BM25 检索
            bm25_results = bm25.search(test["q"], n_results=5)
            print(f"[BM25 检索] 找到 {len(bm25_results)} 条")
            for r in bm25_results[:3]:
                meta = r["metadata"]
                print(
                    f"  · [{r['score']:.3f}] {meta.get('fiscal_period', '?')} | "
                    f"{meta.get('section_title', '')[:40]}"
                )

            # 混合检索
            fused = retriever.retrieve(test["q"])
            print(f"[混合检索] 融合后 {len(fused)} 条")
            for r in fused[:3]:
                meta = r["metadata"]
                print(
                    f"  · [RRF={r.get('rrf_score', r['score']):.4f}] "
                    f"{meta.get('fiscal_period', '?')} | "
                    f"{meta.get('section_title', '')[:40]} | "
                    f"type={meta.get('chunk_type', '?')}"
                )

            # 上下文扩展
            expanded = expander.expand(fused[:5])
            print(f"[上下文扩展] {len(fused[:5])} → {len(expanded)} 个 chunk")

            # 内容预览
            if expanded:
                print(f"\n最相关 chunk 内容预览（前200字）：")
                print(f"  周期: {expanded[0]['metadata'].get('fiscal_period', '?')}")
                print(f"  章节: {expanded[0]['metadata'].get('section_title', '?')}")
                print(f"  内容: {expanded[0]['content'][:200]}...")

            passed += 1
            print(f"\n✅ 测试通过")

        except Exception as e:
            failed += 1
            logger.error(f"❌ 测试失败: {e}", exc_info=True)
            print(f"\n❌ 测试失败: {e}")

    _print_separator("检索测试汇总")
    print(f"通过: {passed}/{len(questions)}  失败: {failed}/{len(questions)}")
    if failed == 0:
        print("🎉 所有检索测试通过！")
    return failed == 0


# ── 完整 QA 模式测试 ───────────────────────────────────────────
def run_qa_tests(questions: List[Dict] = None):
    """
    完整 QA 测试（意图分析 + 检索 + 扩展 + LLM 生成答案）。
    需要设置环境变量 OPENAI_API_KEY（或 LLM_CONFIG 中指定的 api_key_env）。
    """
    from core.config import LLM_CONFIG

    api_key_env = LLM_CONFIG["api_key_env"]
    if not os.environ.get(api_key_env):
        print(f"\n❌ 错误：未设置 LLM API Key 环境变量 '{api_key_env}'")
        print(f"   Windows: set {api_key_env}=sk-...")
        print(f"   Linux/Mac: export {api_key_env}=sk-...")
        sys.exit(1)

    from core.qa.pipeline import TeslaQAPipeline

    _print_separator("初始化 QA Pipeline")
    pipeline = TeslaQAPipeline()  # 自动从 EMBEDDING_CONFIG 加载 embedder

    questions = questions or TEST_QUESTIONS

    passed = 0
    failed = 0

    for i, test in enumerate(questions):
        _print_separator(f"QA 测试 {i+1}/{len(questions)}: {test.get('desc', test['type'])}")
        print(f"问题: {test['q']}")
        print("-" * 65)

        try:
            result = pipeline.ask(test["q"], verbose=True)

            print(f"意图类型:  {result['intent']['type']}")
            print(f"识别年份:  {result['intent']['years']}")
            print(f"识别期间:  {result['intent']['fiscal_periods']}")
            print(f"需要表格:  {result['intent']['needs_table']}")
            print(f"使用 chunk 数: {result['context_count']}")
            print(f"来源引用:  {result['sources'][:3]}")
            if result.get("usage"):
                usage = result["usage"]
                print(
                    f"Token 用量: 输入={usage.get('input_tokens', '?')} "
                    f"输出={usage.get('output_tokens', '?')}"
                )

            print(f"\n── 答案──")
            print(result["answer"])
            # if len(result["answer"]) > 600:
            #     print("... [已截断]")

            passed += 1
            print(f"\n✅ QA 测试通过")

        except Exception as e:
            failed += 1
            logger.error(f"❌ QA 测试失败: {e}", exc_info=True)
            print(f"\n❌ QA 测试失败: {e}")

    _print_separator("QA 测试汇总")
    print(f"通过: {passed}/{len(questions)}  失败: {failed}/{len(questions)}")
    if failed == 0:
        print("🎉 所有 QA 测试通过！")
    return failed == 0


# ── 快速诊断：仅验证模块导入和 ChromaDB 连接 ──────────────────
def run_import_check():
    """快速检查所有模块是否可以正常导入。"""
    _print_separator("模块导入检查")

    checks = [
        ("core.config", ["RETRIEVAL_CONFIG", "LLM_CONFIG", "BM25_INDEX_PATH"]),
        ("core.embedder.factory", ["get_embedder"]),
        ("core.retrieval.vector_store", ["VectorStore"]),
        ("core.retrieval.bm25_retriever", ["BM25Retriever"]),
        ("core.retrieval.hybrid_retriever", ["HybridRetriever", "rrf_fusion"]),
        ("core.retrieval.context_expander", ["ContextExpander"]),
        ("core.qa.query_analyzer", ["QueryAnalyzer", "QueryIntent"]),
        ("core.qa.answer_generator", ["AnswerGenerator"]),
        ("core.qa.pipeline", ["TeslaQAPipeline"]),
    ]

    all_ok = True
    for module_name, symbols in checks:
        try:
            mod = __import__(module_name, fromlist=symbols)
            for sym in symbols:
                assert hasattr(mod, sym), f"缺少 {sym}"
            print(f"  ✅ {module_name}")
        except Exception as e:
            print(f"  ❌ {module_name}: {e}")
            all_ok = False

    # 尝试连接 ChromaDB
    print("\n[ChromaDB 连接测试]")
    try:
        from core.retrieval.vector_store import VectorStore
        vs = VectorStore()
        count = vs.collection.count()
        print(f"  ✅ ChromaDB 连接成功，记录数: {count}")
    except Exception as e:
        print(f"  ❌ ChromaDB 连接失败: {e}")
        all_ok = False

    # 检查 BM25 依赖
    print("\n[BM25 依赖检查]")
    try:
        from rank_bm25 import BM25Okapi
        print("  ✅ rank_bm25 可用")
    except ImportError:
        print("  ❌ rank_bm25 未安装，请运行: pip install rank-bm25")
        all_ok = False

    return all_ok


# ── 入口 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tesla RAG Phase 2 检索与问答测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 仅检查模块导入和 ChromaDB 连接（最快）
  conda run -n tesla python tests/test_retrieval.py --mode check

  # 测试检索流程（不调 LLM）
  conda run -n tesla python tests/test_retrieval.py --mode retrieval

  # 完整 QA 测试（需设置 OPENAI_API_KEY）
  conda run -n tesla python tests/test_retrieval.py --mode qa

  # 测试单条自定义问题
  conda run -n tesla python tests/test_retrieval.py --mode qa --question "2023年特斯拉总营收？"
""",
    )
    parser.add_argument(
        "--mode",
        choices=["check", "retrieval", "qa"],
        default="retrieval",
        help="测试模式：check(导入检查) | retrieval(检索流程) | qa(完整问答)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="单条自定义测试问题（不指定则运行全部预设测试用例）",
    )

    args = parser.parse_args()

    # 自定义单条问题
    questions = None
    if args.question:
        questions = [{"q": args.question, "type": "custom", "desc": "自定义问题"}]

    _print_separator("Tesla RAG Phase 2 测试")

    if args.mode == "check":
        ok = run_import_check()
        sys.exit(0 if ok else 1)
    elif args.mode == "retrieval":
        ok = run_retrieval_tests(questions)
        sys.exit(0 if ok else 1)
    else:  # qa
        ok = run_qa_tests(questions)
        sys.exit(0 if ok else 1)
