import os
import sys
from pathlib import Path
import chromadb

# 加入项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DB_DATA_DIR, RETRIEVAL_CONFIG
from core.retrieval.vector_store import VectorStore
from core.embedder.factory import get_embedder
from core.config import EMBEDDING_CONFIG

def test_chroma_contents():
    print("=== ChromaDB 诊断测试 ===")
    
    try:
        # 直接使用 chromadb API 检查
        client = chromadb.PersistentClient(path=str(DB_DATA_DIR))
        collection_name = RETRIEVAL_CONFIG["chroma_collection_name"]
        
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            print(f"[-] 集合 '{collection_name}' 不存在或无法获取。错误: {e}")
            return
            
        count = collection.count()
        print(f"[+] 成功连接集合 '{collection_name}'，总记录数: {count}")
        
        if count == 0:
            print("[-] 集合为空！块（chunks）没有成功写入数据库。请检查嵌入（embedding）和入库代码。")
            return
            
        # 获取最新的几条记录的元数据，看是否包含 2025Q2 或 2025Q3
        print("\n=== 元数据检查 ===")
        sample = collection.peek(limit=10)
        periods = set()
        for meta in sample.get("metadatas", []):
            if meta and "fiscal_period" in meta:
                periods.add(meta["fiscal_period"])
        print(f"[+] 样本中包含的 fiscal_period: {periods}")
        
        # 尝试使用 filter 直接查找 2025Q2/Q3 的数据
        print("\n=== 过滤器测试 ===")
        q2_results = collection.get(
            where={"$and": [{"year": {"$eq": 2025}}, {"quarter": {"$eq": 2}}]},
            limit=5
        )
        print(f"[+] 匹配 year=2025, quarter=2 (2025Q2) 的记录数: {len(q2_results.get('ids', []))}")
        
        q3_results = collection.get(
            where={"$and": [{"year": {"$eq": 2025}}, {"quarter": {"$eq": 3}}]},
            limit=5
        )
        print(f"[+] 匹配 year=2025, quarter=3 (2025Q3) 的记录数: {len(q3_results.get('ids', []))}")
        q3_results = collection.get(
            where={"$and": [{"fiscal_period": {"$eq": "2025Q1"}}, {"report_type": {"$eq": "10-Q"}}]},
            limit=5
        )
        print(f"[+] 匹配 fiscal_period=2025Q1, report_type=10-Q (2025Q1) 的记录数: {len(q3_results.get('ids', []))}")
        
        if len(q2_results.get('ids', [])) == 0 and len(q3_results.get('ids', [])) == 0:
            print("\n[-] 警告：数据库中存在数据，但无法通过 2025Q2/Q3 的元数据过滤找到它们。")
            print("    可能原因：")
            print("    1. 2025Q2_chunks.json 和 2025Q3_chunks.json 确实没有入库。")
            print("    2. 入库时，metadata 字段类型被改变了（例如 year 被存成了字符串 '2025' 而不是数字 2025）。")
            
            # 取出一条看看它的 metadata 长什么样
            all_meta = collection.get(limit=1, include=["metadatas"])
            print(f"\n[+] 数据库中第一条记录的元数据样例:\n{all_meta.get('metadatas', [None])[0]}")

        else:
            print("\n[+] metadata 过滤正常！下面使用向量检索测试完整的检索流程...")
            
            embedder = get_embedder(EMBEDDING_CONFIG)
            query = "比较2025年 Q2 和 Q3 财报中关于“One Big Beautiful Bill Act (OBBBA)”对特斯拉业务影响的描述"
            query_vec = embedder.embed_one(query)
            
            vs = VectorStore()
            
            print("\n=== VectorStore 检索测试 (带 Q2 过滤器) ===")
            results_q2 = vs.query(
                query_embedding=query_vec, 
                n_results=5, 
                where={"$and": [{"year": {"$eq": 2025}}, {"quarter": {"$eq": 2}}]}
            )
            print(f"找到 Q2 结果数量: {len(results_q2)}")
            for r in results_q2[:2]:
                print(f"  - [{r['score']:.4f}] {r['metadata'].get('chunk_id')} | {r['content'][:50]}...")

            print("\n=== VectorStore 检索测试 (带 Q3 过滤器) ===")
            results_q3 = vs.query(
                query_embedding=query_vec, 
                n_results=5, 
                where={"$and": [{"year": {"$eq": 2025}}, {"quarter": {"$eq": 3}}]}
            )
            print(f"找到 Q3 结果数量: {len(results_q3)}")
            for r in results_q3[:2]:
                print(f"  - [{r['score']:.4f}] {r['metadata'].get('chunk_id')} | {r['content'][:50]}...")

    except Exception as e:
        print(f"发生异常: {e}")

if __name__ == "__main__":
    test_chroma_contents()
