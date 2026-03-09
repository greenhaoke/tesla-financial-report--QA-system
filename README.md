# Tesla Financial Reports RAG System

本项目是一个专门针对特斯拉（Tesla）财务报告（10-K 和 10-Q）构建的 Retrieval-Augmented Generation (RAG) 系统。该系统旨在通过高级解析、混合检索和大型语言模型（LLM）的深度推理，准确保留并回答结构繁杂、充满数字及上下文跳跃的财务询问。

## 🚀 运行指南

### 1. 环境准备
确保已安装 Python 3.10+，并在项目根目录下安装所有依赖：
```bash
pip install -r requirements.txt
```

### 2. 设置配置
修改core/config.py，设置api_key和api_model，以及检索的配置

### 3. 启动系统交互界面 (Streamlit)
项目提供了一个直观的 Web UI 来进行财报查询：
```bash
streamlit run app/main.py
```
在浏览器中打开提供的本地地址，即可选择“查询范围（报告类型）”，输入问题，并实时查看模型给出的详细答案、引用片段和 Debug 检索信息。
在web.png中可以看到web运行结果

### 4. (可选) 重新构建本地向量库与索引
如需重新解析和灌库，需要先运行 `test_parser.py` 来解析 PDF 为分块文件，再运行 `embed_pipeline.py` 将其嵌入并写入数据库。

**步骤 4.1: 解析分块**  
解析指定 PDF 文件（参数设置在 `test_parser.py` 中，通常会将生成的 chunk 输出到 `output` 目录下）：
```bash
python test_parser.py --pdf data/10-K/tesla_10k_2025.pdf --step all
```
*   `--pdf`: 指定要解析的原始 PDF 文件路径。
*   `--step`: 可选在解析生命周期中运行哪些步（`parser`, `chunker`, `pipeline`, `all`）。

**步骤 4.2: 向量化嵌入 (Embedding)**  
运行嵌入流水线（支持目录批量处理或单文件，默认参数通过 `core/config.py` 中的 `EMBEDDING_CONFIG` 读取，也可以通过命令行覆盖）：
```bash
# 批量处理 output/ 目录下所有 JSON（本地模型自动检测 GPU）
python core/embedder/embed_pipeline.py --json-dir output --provider local

# 处理单个分块文件并使用 API 嵌入
python core/embedder/embed_pipeline.py --json-file output/FY2025_chunks.json --provider api --api-model text-embedding-3-small
```
*   `--json-dir / --json-file`: 指定分块文件所在的目录或单文件路径。
*   `--provider`: 选择嵌入方式（`local` 使用本地 sentence-transformers 模型，`api` 使用兼容 OpenAI 接口的模型）。
*   `--force`: 强制重新嵌入已存在的 id（跳过缓存记录）。

---

## 📊 数据概览

本系统实际成功解析、分块（Chunking）处理并存入图谱（ChromaDB + BM25）的财报数据涵盖：
*   **年度报告 (10-K)**：FY2023, FY2024, FY2025
*   **季度报告 (10-Q)**：2025 Q1, 2025 Q2, 2025 Q3

---

## 🏗️ 系统设计抉择

针对财务报表（SEC Filings）中数据密集、长篇幅附注（Notes）极多且跨年度变化难以追踪的痛点，我们在系统设计中做出了如下核心抉择：

1.  **解析策略：Deepdoc OCR 与混合表切割**
    单纯的文本抽取（如 PyPDF2）极易丢失表格结构。我们采用了ragflow开发的深度文档解析器（Deepdoc），尝试在保留表格行列逻辑的前提下进行分块，以解决资产负债表等复杂二维矩阵的抽取痛点。
2.  **检索策略：ChromaDB Vector + BM25 混合双路**
    财务检索中很多缩写、法案名（如 OBBBA）具有唯一的强字面特征。单纯的稠密向量（Dense Vector）检索很容易错过这些稀疏特征。因此，采用了基于 ChromaDB 的向量检索加上 BM25 字词匹配的混合召回（Hybrid Retrieval），结合倒数秩融合（RRF）算法，兼顾了语义理解与字面精准命中。
3.  **上下文衔接：Small2Big Context Expansion (按期平摊)**
    财报存在极强的上下文依赖（单独看“下降20%”无法知道标的）。系统采用了“小块比对，大块送入”的策略（Small-to-Big Retrieval），检索时匹配句子粒度的隐式信息，但在最终送入大模型前自动扩展并找回其前后的完整段落或表格。特别加入了 **“按期平摊（Per-Period Expansion）”** 逻辑，强制跨年对比时均衡保留各年份数据。
4.  **排序策略：LLM Reranker**
    引入轻量级的 LLM 进行最终的 1-to-N 相似度打分（LLM Reranker），利用大模型的深层财务理解能力剔除高向量相似但实质为错配噪音的片段，确保最终送去生成答案的 Token 池绝对高纯度。
5.  **元数据硬过滤（Metadata Filtering）**
    基于财报年份、季度、报告类型（10-K/10-Q）实施硬过滤（通过大模型 `QueryAnalyzer` 自动解析出 ChromaDB DB 过滤语法），实现前端 UI 的文档范围精准框选，从源头切断“张冠李戴”式的幻觉。

---

## 📝 测试集与结果摘要

我们在由 14 个高难度问题组成的测试集上对系统进行了验证，评价结果见下表：

| 测试题号 | 考核考点 | 结果评价 | 
| :--- | :--- | :--- |
| **Q1** | 政府激励措施政策对比 (OBBBA 法案) | ✅ 成功 |
| **Q2** | 埃隆·马斯克任职公司与政府状态跨年对比 | ✅ 成功 |
| **Q3** | 2024 与 2025 会计年度重组支出金额与动因 | ❌ 失败 |
| **Q4** | 资本支出 (CapEx) 的跨期预测变化追踪 | ✅ 成功 |
| **Q5** | 2018 年 CEO 薪酬案的法律时间线追踪 | ✅ 成功 |
| **Q6** | OBBBA 对细分业务与风险的具体担忧 | ❌ 失败 |
| **Q7** | 汽车业务毛利同比最差季度的数值提取与归因 | ✅ 成功 |
| **Q8** | Robotaxi 业务首秀时间表提取 | ✅ 成功 |
| **Q9** | 监管信用额度波动的口径跨季翻转 (Q1 vs Q2) | ✅ 成功 |
| **Q10** | 能源业务毛利提升与绝对营收数字的勾稽关系推理 | ⚠️ 部分成功 |
| **Q11** | 资产负债表“成品库存”提取及关联 MD&A 交付事故 | ❌ 失败 |
| **Q12** | 经营活动净现金流变动的核心报表项目调节项解释 | ✅ 成功 |
| **Q13** | Q1-Q3 研发费用率峰值比较及除 AI 项目外的 Q3 原因 | ❌ 失败 |
| **Q14** | 持有比特币数量及 Q1 vs Q2 的名义市场损益分辨 | ⚠️ 部分成功 |

---

## 🔍 失败案例深度剖析

针对上述表格中表现欠佳（失败或部分成功）的典型案例（Q3, Q6, Q10, Q11, Q13），我们进行了深度溯源和归因。

核心发现表明：标准 RAG 流程在此类专业财务用例中遇到了**长篇幅财务附注截断导致权重衰减**、**复杂表格数字关联性断裂**、以及**“跨期公平性”算法被噪音底层召回冲平**三大瓶颈。

请参阅项目中专门梳理的分析长文文档：
👉 **[FAILURE_ANALYSIS.md](./FAILURE_ANALYSIS.md)** 了解我们的完整归因排查以及针对下个迭代的破局提议（例如启用父子图谱树索引结构）。


## 文件目录结构

tesla/
├── app/                          # 🖥️ Streamlit Web 交互界面
├── core/                         # 🧠 核心业务逻辑与 RAG 组件
│   ├── api/                      #    deepdo所需要用到的包
│   ├── deepdoc/                  #    RagFlow 深度文档解析引擎
│   │   ├── parser/               #       格式分析与区块提取
│   │   └── vision/               #       OCR 视觉识别模型
│   ├── embedder/                 #    向量化流水线 (支持 Local/API)
│   ├── parser/                   #    文档处理与语义分块 (Chunking)
│   ├── qa/                       #    意图分析与回答生成 (QA Pipeline)
│   ├── rag/                      #    deepdo所需要用到的包
│   ├── retrieval/                #    混合检索、重排序与上下文平摊拓展
│   └── utils/                    #    通用工具与辅助函数库
├── data/                         # 📂 原始财报 PDF 数据存放处
│   ├── 10-K/                     #    年度报表数据 (Annual)
│   └── 10-Q/                     #    季度报表数据 (Quarterly)
├── db_data/                      # 🗄️ 本地构建的数据库与检索索引
│   └── chroma/                   #    ChromaDB 向量数据持久卷
├── output/                       # 📦 PDF 解析后生成的 JSON Chunk 产物
├── FAILURE_ANALYSIS.md           # ⚠️ 5 大典型失败案例核心归因深度剖析
├── README.md                     # 📖 项目运行指南及设计总体说明文档
├── log.txt                       # 📝 核心模块执行日志与查询链路追踪
├── requirements.txt              # 📦 Python 第三方包依赖环境清单
├── results.txt                   # 📊 RAG 测试用例集批量执行生成的详细评估
├── test_chromadb_diagnostics.py  # 🛠️ ChromaDB 诊断修复独立脚本
├── test_embedding.py             # 🛠️ 向量化 Embedding 流水线测试脚本
├── test_parser.py                # 🛠️ 各种格式 Document Parsing/Chunking 测试验证
├── test_retrieval.py             # 🛠️ 整体 RAG Retriever 和 QA 链路测试
└── web.png                       # 🖼️ 提供预览的 Web UI 系统截图
