import streamlit as st
import sys
import os
from pathlib import Path

# 确保项目根目录在 python path 中，以便正常导入 core 模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.qa.pipeline import TeslaQAPipeline
from core.embedder.factory import get_embedder
from core.config import EMBEDDING_CONFIG

st.set_page_config(
    page_title="特斯拉财报智能问答系统",
    page_icon="🚗",
    layout="wide"
)

# ====== 全局初始化 ======
@st.cache_resource
def get_pipeline():
    # 首次加载缓存 pipeline (包含大模型加载)
    with st.spinner("正在加载 Embedding 大模型和问答引擎，这可能需要一点时间..."):
        embedder = get_embedder(EMBEDDING_CONFIG)
        return TeslaQAPipeline(embedder=embedder)

pipeline = get_pipeline()

# ====== 界面布局 ======
st.title("🚗 特斯拉财报智能问答系统 (RAG)")
st.markdown("基于 **2023-2025** 年特斯拉财报（10-K / 10-Q）内容的向量检索与智能解答。")

# 侧边栏：检索设置
with st.sidebar:
    st.header("⚙️ 检索设置")
    scope_option = st.selectbox(
        "查询文档范围",
        options=["所有文档", "仅 10-K (年报)", "仅 10-Q (季报)"],
        index=0,
        help="限制检索只在特定类型的财报中进行。"
    )
    
    show_debug = st.checkbox("显示检索调试信息", value=False)
    
    # 将选项转换为我们传递给 pipeline 的参数
    report_type_filter = None
    if scope_option == "仅 10-K (年报)":
        report_type_filter = "10-K"
    elif scope_option == "仅 10-Q (季报)":
        report_type_filter = "10-Q"

# ====== 问答对话区 ======
# 使用 session_state 保存对话记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "references" in msg and msg["references"]:
            with st.expander("参考片段 (点击展开)"):
                for ref in msg["references"]:
                    st.markdown(ref)
        if "debug_info" in msg and msg["debug_info"] and show_debug:
            with st.expander("调试信息：意图与检索统计"):
                st.json(msg["debug_info"])

# 接收用户输入
user_query = st.chat_input("询问特斯拉财报信息（例如：2023年总营收是多少？）")

if user_query:
    # 1. 立即显示用户的问题
    with st.chat_message("user"):
        st.markdown(user_query)
        
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. 调用 Pipeline 生成答案
    with st.chat_message("assistant"):
        with st.spinner("思考与检索中..."):
            try:
                # 调用问答管道
                result = pipeline.ask(
                    question=user_query,
                    verbose=True, 
                    report_type=report_type_filter
                )
                
                answer_text = result.get("answer", "未能生成答案。")
                sources = result.get("sources", [])
                chunks = result.get("retrieved_chunks", [])
                intent = result.get("intent", {})
                
                # 展示答案
                st.markdown(answer_text)
                
                # 展示参考片段
                references = []
                if chunks:
                    st.markdown("---")
                    st.markdown("**🔍 引用来源**")
                    for i, chunk in enumerate(chunks):
                        meta = chunk.get("metadata", {})
                        p = meta.get("fiscal_period", "Unknown")
                        s = meta.get("section_title", "Unknown")
                        t = meta.get("chunk_type", "Unknown")
                        score = chunk.get("score", 0.0)
                        
                        ref_title = f"[{i+1}] {p} | {s} | {t} (相关度: {score:.3f})"
                        ref_content = chunk.get("content", "")
                        
                        references.append(f"**{ref_title}**\n\n```text\n{ref_content}\n```")
                    
                    with st.expander("参考片段 (点击展开)"):
                        for ref in references:
                            st.markdown(ref)
                
                # 记录调试信息
                debug_info = {
                    "Intent": intent,
                    "Retrieved Chunks Count": len(chunks),
                    "Context Count Used": result.get("context_count", 0),
                    "Token Usage": result.get("usage", {})
                }
                
                if show_debug:
                    with st.expander("调试信息：意图与检索统计"):
                        st.json(debug_info)
                
                # 存入历史以备后续渲染
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "references": references,
                    "debug_info": debug_info
                })
                
            except Exception as e:
                st.error(f"处理时出错: {str(e)}")
