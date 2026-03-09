# core/qa/answer_generator.py
"""
答案生成器：将检索到的上下文 + 问题送入 LLM（OpenAI 兼容接口）生成最终答案。

支持：
- 带来源引用的结构化答案
- 数字计算的分步推理
- 对比类问题的表格化输出
"""
import os
import logging
from typing import List, Dict, Any

from openai import OpenAI

from core.config import LLM_CONFIG

logger = logging.getLogger(__name__)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """将 chunk 列表格式化为 LLM 可读的上下文字符串。"""
    parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        period = meta.get("fiscal_period", "Unknown")
        section = meta.get("section_title", "")
        chunk_type = meta.get("chunk_type", "text")
        page = meta.get("page_start", "?")

        header = f"[来源 {i+1}] {period} | {section} | 第{page}页"
        if chunk_type == "table":
            table_title = meta.get("table_title", "")
            if table_title:
                header += f" | 表格: {table_title}"

        parts.append(f"{header}\n{chunk['content']}")

    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """你是一位专业的特斯拉财报分析师，拥有深厚的财务知识。
你的任务是基于提供的财报原文片段，准确、客观地回答关于特斯拉财务状况的问题。

回答规范：
1. **严格基于原文**：只使用提供的原文片段中的信息，不要凭记忆补充未提供的数据
2. **明确引用来源**：每个关键数据或结论后，用 [来源N: 期间 | 章节] 格式注明出处
3. **数字计算要展示过程**：如需计算环比/同比，写出计算公式和中间步骤
4. **区分文本和数据**：财务数据来自表格时说明，文字描述来自 MD&A 等章节时说明
5. **信息不足时明确说明**：如果提供的片段不足以回答某部分问题，明确指出"原文中未找到相关信息"
6. **结构化输出**：复杂问题用分点回答，对比问题优先用 Markdown 表格呈现"""

ANSWER_PROMPT = """请基于以下特斯拉财报原文片段，回答用户的问题。

## 财报原文片段

{context}

## 用户问题

{question}

## 回答要求

{special_instructions}

请现在给出详细、准确的分析答案："""


def _get_special_instructions(question_type: str, needs_table: bool) -> str:
    instructions = {
        "single_point": "直接给出答案，注明数据来源的期间和章节。",
        "comparison": "用 Markdown 表格对比不同期间的数据，表格后加文字分析。",
        "trend": "按时间顺序列出数据点，分析趋势方向和原因。",
        "calculation": "明确写出计算公式、代入数值、得出结果，如有单位换算请注明。",
        "multi_step": "按子问题逐步回答，最后综合得出结论。",
    }
    base = instructions.get(question_type, "详细回答问题，注明来源。")
    if needs_table:
        base += "\n注意：优先使用表格数据中的精确数字，文本中的描述作为补充。"
    return base


class AnswerGenerator:
    """使用 LLM（OpenAI 兼容接口）生成最终答案。"""

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
        self.model = LLM_CONFIG["model"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.temperature = LLM_CONFIG["temperature"]

    def generate(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        question_type: str = "single_point",
        needs_table: bool = False,
    ) -> Dict[str, Any]:
        """
        生成答案。

        Returns:
            {
                "answer": str,           # 最终答案文本
                "sources": list,         # 引用的来源列表
                "model": str,            # 使用的模型
                "context_count": int,    # 使用的 chunk 数量
                "usage": dict,           # token 用量
            }
        """
        if not context_chunks:
            return {
                "answer": "抱歉，在财报中未找到与该问题相关的内容。请确认查询的时间范围和关键词是否正确。",
                "sources": [],
                "model": self.model,
                "context_count": 0,
                "usage": {},
            }

        context_str = _format_context(context_chunks)
        special_inst = _get_special_instructions(question_type, needs_table)

        prompt = ANSWER_PROMPT.format(
            context=context_str,
            question=question,
            special_instructions=special_inst,
        )

        logger.info(
            f"[AnswerGenerator] 生成答案，使用 {len(context_chunks)} 个 chunk，"
            f"模型: {self.model}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        answer_text = response.choices[0].message.content

        # 提取引用来源（去重）
        sources = sorted({
            f"{c['metadata'].get('fiscal_period', 'Unknown')} | "
            f"{c['metadata'].get('section_title', '')}"
            for c in context_chunks
        })

        return {
            "answer": answer_text,
            "sources": sources,
            "model": self.model,
            "context_count": len(context_chunks),
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
