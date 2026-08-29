from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
import os

app = FastAPI(title="优高Demo后端")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管静态前端文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 根路径直接返回首页
@app.get("/")
async def root():
    return FileResponse("static/index.html")

class ChatRequest(BaseModel):
    question: str
    scene: str = "default"

# 模拟知识库
KNOWLEDGE = {
    "优高系列": "优高系列包含三款产品：优高智能（AI-Agent底座）、优高智人（数字人交互）、优高智境（AI+XR空间）。三者原生打通，形成完整的AI虚实融合闭环。",
    "优高智能": "优高智能是产业级AI-Agent底座，核心能力包括：RAG知识库、多智能体编排、业务流程自动化、开放API网关。作为整个体系的大脑，为上层应用提供AI能力支撑。",
    "优高智人": "优高智人是人格化数字交互平台，支持自定义数字人形象、音色、人格设定，实现语音对话、表情唇形同步。既可独立使用，也可嵌入XR空间。",
    "优高智境": "优高智境是AI+XR虚实融合空间平台，支持三维场景编辑、多终端访问、空间事件驱动AI交互。覆盖虚拟展厅、工业实训、数字园区等场景。",
    "技术架构": "整体采用分层架构：基础设施层→通用能力中台→统一网关层→应用层。共用底座能力，三款产品原生互通，支持独立部署或一体化交付。",
    "应用场景": "典型应用场景包括：政企虚拟展厅、工业技能实训、数字园区展示馆、文旅虚拟导览、企业智能知识中枢等。"
}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """模拟AI问答接口"""
    time.sleep(0.5)
    
    answer = "抱歉，知识库中暂未找到相关内容。您可以询问关于优高系列产品、技术架构、应用场景等问题。"
    source = "无匹配结果"
    
    for keyword, content in KNOWLEDGE.items():
        if keyword in request.question:
            answer = content
            source = f"知识库 · {keyword}相关文档"
            break
    
    return {
        "answer": answer,
        "source": source,
        "timestamp": int(time.time())
    }

@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "优高Demo后端运行中"}

if __name__ == "__main__":
    # 本地运行用，部署时平台会自动指定端口
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)