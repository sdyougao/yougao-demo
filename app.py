from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import time
import requests
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="优高智能 RAG 演示站")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管静态前端
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ===================== 配置读取 =====================
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-701d6b0a4f8443e996eb1bbd086498b5")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v2")
TOP_K = int(os.environ.get("TOP_K", 3))

# ===================== 向量库初始化 =====================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="yougao_demo",
    metadata={"description": "优高智能演示知识库"}
)

# ===================== Embedding 函数 =====================
def get_embedding(text: str):
    """调用大模型 Embedding 接口生成向量"""
    if not LLM_API_KEY:
        # 未配置API时降级为简单关键词匹配
        return None
    
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding 调用失败: {e}")
        return None

# ===================== 预置演示知识库 =====================
DEMO_DOCUMENTS = [
    {
        "title": "优高系列产品白皮书V2.0",
        "content": """优高系列包含三款产品：优高智能、优高智人、优高智境。
优高智能是AI-Agent产业智能底座，核心能力包括RAG智能知识库、多智能体编排、业务流程自动化、开放API网关。作为整个体系的大脑，为上层应用提供AI能力支撑。
优高智人是人格化数字交互平台，支持自定义数字人形象、音色、人格设定，实现语音对话、表情唇形同步。既可独立使用，也可嵌入XR空间。
优高智境是AI+XR虚实融合空间平台，支持三维场景编辑、多终端访问、空间事件驱动AI交互。覆盖虚拟展厅、工业实训、数字园区等场景。
三款产品原生打通，形成完整的AI虚实融合闭环，支持独立部署或一体化交付。"""
    },
    {
        "title": "技术架构说明文档",
        "content": """优高智能整体采用分层解耦架构，自上而下分为应用层、统一网关层、通用能力中台层、基础设施层。
应用层包括优高智能管理后台、优高智人数字交互、优高智境XR空间。
统一网关层提供API网关、身份认证、权限校验、流量管控能力。
通用能力中台层包含RAG知识库引擎、Agent编排引擎、多模态引擎、权限审计中心。
基础设施层包含计算资源、容器编排、向量与关系数据库、安全防护体系。
支持三种部署模式：SaaS公有云模式、私有化部署模式、混合云模式。全面适配信创环境，支持国产CPU、操作系统、数据库与大模型。"""
    },
    {
        "title": "政企展厅解决方案",
        "content": """政企智能展厅方案采用优高三位一体架构，为展厅数字人、XR空间提供业务知识库与智能交互能力。
传统展厅只能被动浏览，方案升级后可实现：虚拟讲解员主动迎宾、展品智能问答、业务咨询解答、客户线索自动留存。
方案优势：7×24小时不间断开放，突破实体展厅时空限制；虚拟讲解员替代人工接待，降低运营人力成本；从被动浏览升级为可问答、可留资、可转化的业务型展厅。
支持PC、大屏、VR多端访问，适配不同参观场景。"""
    },
    {
        "title": "工业实训场景方案",
        "content": """工业技能实训方案结合XR实训场景与AI知识库，实现智能考核与个性化指导。
解决高危高成本岗位实操培训风险大、设备损耗高、培训名额受限的痛点。
学员在VR场景中进行模拟操作，每一步动作由AI实时校验是否符合操作规程，操作错误时虚拟教练即时提示纠正。
实训结束后自动生成个人考核报告，包括操作正确率、耗时、易错环节、知识薄弱点。
实现零风险零损耗反复演练，大幅降低实训成本与安全隐患，提升学员记忆留存率与实操熟练度。"""
    }
]

# 启动时自动构建知识库
def init_knowledge_base():
    """初始化演示知识库，已存在则跳过"""
    if collection.count() > 0:
        print(f"知识库已存在，共 {collection.count()} 条文档")
        return
    
    print("正在构建演示知识库...")
    documents = []
    metadatas = []
    ids = []
    
    for i, doc in enumerate(DEMO_DOCUMENTS):
        documents.append(doc["content"])
        metadatas.append({"title": doc["title"]})
        ids.append(f"doc_{i}")
    
    # 无API时用默认嵌入（仅演示）
    if not LLM_API_KEY:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    else:
        # 批量生成向量
        embeddings = []
        for doc in documents:
            emb = get_embedding(doc)
            embeddings.append(emb if emb else [0.0]*1024)
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
    
    print(f"知识库构建完成，共 {len(documents)} 条文档")

# 应用启动时执行
@app.on_event("startup")
async def startup_event():
    init_knowledge_base()

# ===================== RAG 核心逻辑 =====================
def retrieve_relevant_docs(question: str):
    """检索相关文档"""
    query_embedding = get_embedding(question)
    
    if query_embedding:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K
        )
    else:
        # 降级：关键词检索
        results = collection.query(
            query_texts=[question],
            n_results=TOP_K
        )
    
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "content": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "distance": results["distances"][0][i] if "distances" in results else 0
        })
    return docs

def generate_answer(question: str, docs: list):
    """调用大模型生成 RAG 答案"""
    if not LLM_API_KEY:
        # 未配置API时降级返回拼接结果
        answer = "【演示模式 - 未接入真实大模型】\n\n"
        answer += f"针对问题「{question}」，检索到以下相关知识：\n\n"
        for doc in docs:
            answer += f"### 出自《{doc['title']}》\n{doc['content'][:200]}...\n\n"
        answer += "配置 LLM_API_KEY 环境变量后即可启用真实大模型生成。"
        return answer, docs[0]["title"] if docs else "无匹配结果"
    
    # 构造 RAG Prompt
    context = "\n\n".join([
        f"【文档：{d['title']}】\n{d['content']}" 
        for d in docs
    ])
    
    prompt = f"""你是优高智能的官方助手，基于以下参考资料回答用户的问题。
要求：
1. 只使用参考资料中的信息，不要编造内容
2. 回答简洁专业，符合B端产品风格
3. 如果参考资料中没有答案，直接说明暂未找到相关内容
4. 不要提及"参考资料"、"文档"等表述

参考资料：
{context}

用户问题：{question}
"""
    
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        source = "、".join([d["title"] for d in docs])
        return answer, source
    
    except Exception as e:
        return f"大模型调用失败：{str(e)}", "系统错误"

# ===================== 接口定义 =====================
class ChatRequest(BaseModel):
    question: str
    scene: str = "default"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """RAG 智能问答接口"""
    # 1. 检索相关文档
    relevant_docs = retrieve_relevant_docs(request.question)
    
    # 2. 生成答案
    answer, source = generate_answer(request.question, relevant_docs)
    
    return {
        "answer": answer,
        "source": f"知识库 · {source}",
        "timestamp": int(time.time())
    }

@app.get("/api/health")
async def health():
    return {
        "status": "ok", 
        "llm_connected": bool(LLM_API_KEY),
        "docs_count": collection.count(),
        "message": "优高智能 RAG 服务运行中"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
