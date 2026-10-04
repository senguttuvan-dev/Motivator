# ---------------------------------------------------------------
# Standalone Motivation Agent App (app.py)
# ---------------------------------------------------------------
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="Motivation Agent API", version="1.0.0")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
system_prompt = (
    "You are an empathetic, highly motivating AI companion. "
    "When a user shares a negative, pessimistic, or demotivating quote/thought, "
    "listen with empathy, gently reframe their perspective, and respond with "
    "an uplifting, encouraging, and motivating message."
)
memory = MemorySaver()
agent_executor = create_react_agent(model=llm, tools=[], prompt=system_prompt, checkpointer=memory)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    latency_ms: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        result = agent_executor.invoke({"messages": [("user", request.message)]}, config=config)
        ai_message = result["messages"][-1].content
        latency = round((time.time() - start_time) * 1000, 2)
        return ChatResponse(session_id=request.session_id, response=ai_message, latency_ms=latency)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
