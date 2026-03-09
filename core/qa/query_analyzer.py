# core/qa/query_analyzer.py
"""
查询意图分析器：使用 LLM（OpenAI 兼容接口）将自然语言问题解析为结构化检索意图。

输出的结构化意图将直接驱动 HybridRetriever 的检索参数。
"""
import json
import re
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from openai import OpenAI

from core.config import LLM_CONFIG

logger = logging.getLogger(__name__)

QUERY_ANALYSIS_PROMPT = """你是一个专业的金融文档检索助手。请分析以下问题，提取检索所需的结构化信息。

问题：{question}

请以 JSON 格式返回分析结果，字段说明如下：
- question_type: 问题类型，必须是以下之一：
  * "single_point"  → 询问某个具体事实（如"2022年Q3营收是多少"）
  * "comparison"    → 跨时间/文档对比（如"对比2021-2023年毛利率"）
  * "trend"         → 趋势分析（如"自2020年以来的增长趋势"）
  * "calculation"   → 需要数学计算（如"环比增长率"）
  * "multi_step"    → 需要多步推理（如"哪个季度提到了X，且该季度Y变化如何"）
- years: 涉及的年份列表，如 [2022, 2023]，不确定则为 []
- quarters: 涉及的季度列表，如 [3]，年报问题则为 []
- report_types: 涉及的报告类型列表，可含 "10-K", "10-Q"，不限制则为 []
- fiscal_periods: 涉及的财务期间标签，如 ["FY2022", "2022Q3"]，用于跨文档对比检索
- financial_metrics: 涉及的财务指标关键词列表，如 ["automotive gross margin", "revenue"]
- intent_keywords: 检索关键词列表（最重要的 3-5 个词组），如 ["supply chain", "gross margin"]
- needs_table: 布尔值，问题是否需要表格数据（含数字计算、财务数据对比时为 true）
- needs_text: 布尔值，问题是否需要文本叙述（含"管理层讨论"、"原因"、"背景"时为 true）
- sub_questions: 如果是 multi_step 类型，拆解为子问题列表；其他类型为 []
- metadata_filter: ChromaDB where 过滤条件（JSON格式）。当问题涉及指定的财报期间（如 Q1 和 Q2 或者多个年份），强烈建议在 metadata_filter 中使用 `$or` 来组合条件，因为一个块不可能同时属于两个期间！且 ChromaDB 的 `$and` 和 `$or` 列表中，**每个条件字典必须只包含一个键值对**，绝不能包含多个（如 `{{"year": 2025, "quarter": 1}}` 是非法的，必须拆分为 `{{"$and": [{{"year": {{"$eq": 2025}}}}, {{"quarter": {{"$eq": 1}}}}]}}`）。如果没有约束则为 null。

注意事项：
- 年报（10-K）的 quarter 在数据库中存储为 -1，请勿在 metadata_filter 中使用 null
- fiscal_periods 格式：年报用 "FY2022"，季报用 "2022Q3"

示例输出：
{{
  "question_type": "multi_step",
  "years": [2022],
  "quarters": [],
  "report_types": ["10-Q"],
  "fiscal_periods": ["2022Q1", "2022Q2", "2022Q3"],
  "financial_metrics": ["revenue", "quarter-over-quarter"],
  "intent_keywords": ["supply chain", "challenges", "revenue"],
  "needs_table": true,
  "needs_text": true,
  "sub_questions": [
    "2022年哪份季报提到了供应链挑战？",
    "该季度的营收是多少？",
    "上一季度的营收是多少？",
    "计算环比变化率"
  ],
  "metadata_filter": {{"$and": [{{"year": {{"$eq": 2022}}}}, {{"report_type": {{"$eq": "10-Q"}}}}]}}
}}

只返回 JSON，不要有其他文字。"""


@dataclass
class QueryIntent:
    question_type: str
    years: List[int]
    quarters: List[int]
    report_types: List[str]
    fiscal_periods: List[str]
    financial_metrics: List[str]
    intent_keywords: List[str]
    needs_table: bool
    needs_text: bool
    sub_questions: List[str]
    metadata_filter: Optional[Dict]
    raw_question: str


def _sanitize_chromadb_filter(filter_dict: Optional[Dict]) -> Optional[Dict]:
    """修复 LLM 生成的非法 ChromaDB filter 结构（例如多个 key 在一个 dict 中）。"""
    if not filter_dict or not isinstance(filter_dict, dict):
        return filter_dict

    def _process_node(node):
        if not isinstance(node, dict):
            return node
            
        new_node = {}
        for k, v in node.items():
            if k in ["$and", "$or"] and isinstance(v, list):
                processed_list = []
                for item in v:
                    processed_item = _process_node(item)
                    if isinstance(processed_item, dict):
                        # 如果单个字典有多个字段（且不是包含操作符的字典），需将其拆分并用 $and 包装（即使父级是 $or 或 $and）
                        if len(processed_item) > 1 and not any(k2.startswith('$') for k2 in processed_item.keys()):
                            and_items = [{k2: v2} for k2, v2 in processed_item.items()]
                            processed_list.append({"$and": and_items})
                        else:
                            processed_list.append(processed_item)
                    else:
                        processed_list.append(processed_item)
                new_node[k] = processed_list
            else:
                new_node[k] = _process_node(v) if isinstance(v, dict) else v
                
        # 处理顶级多字段字典（不在 $and/$or 内）
        if len(new_node) > 1 and not any(k.startswith('$') for k in new_node.keys()):
            return {"$and": [{k: v} for k, v in new_node.items()]}
            
        return new_node

    return _process_node(filter_dict)


class QueryAnalyzer:
    """使用 LLM（OpenAI 兼容接口）分析查询意图。"""

    def __init__(self):
        api_key = os.environ.get(LLM_CONFIG["api_key_env"])
        if not api_key:
            raise ValueError(
                f"LLM API Key 未配置，请设置环境变量: {LLM_CONFIG['api_key_env']}"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=LLM_CONFIG["api_base_url"],
        )
        self.fast_model = LLM_CONFIG["fast_model"]

    def analyze(self, question: str) -> QueryIntent:
        """分析问题，返回结构化意图。"""
        logger.info(f"[QueryAnalyzer] 分析查询意图: {question[:60]}...")

        prompt = QUERY_ANALYSIS_PROMPT.format(question=question)

        response = self.client.chat.completions.create(
            model=self.fast_model,
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "你是专业的金融文档检索分析助手，只返回 JSON 格式的分析结果。",
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw_text = response.choices[0].message.content.strip()

        # 提取 JSON（防止 LLM 包裹 ```json ... ``` 或额外文字）
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            raw_text = match.group()

        data = json.loads(raw_text)
        intent = QueryIntent(
            question_type=data.get("question_type", "single_point"),
            years=data.get("years", []),
            quarters=data.get("quarters", []),
            report_types=data.get("report_types", []),
            fiscal_periods=data.get("fiscal_periods", []),
            financial_metrics=data.get("financial_metrics", []),
            intent_keywords=data.get("intent_keywords", []),
            needs_table=data.get("needs_table", False),
            needs_text=data.get("needs_text", True),
            sub_questions=data.get("sub_questions", []),
            metadata_filter=_sanitize_chromadb_filter(data.get("metadata_filter")),
            raw_question=question,
        )

        logger.info(
            f"[QueryAnalyzer] 意图分析完成: type={intent.question_type}, "
            f"years={intent.years}, periods={intent.fiscal_periods}"
        )
        return intent
