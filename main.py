from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import gemini

app = FastAPI(title="EkoBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    answer: str
    is_followup: bool = False

@app.get("/")
def root():
    return {"message": "EkoBot is running!"}

@app.post("/chat")
async def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]

    # Some clients include the current question in the history payload.
    # The backend already receives it separately as `question`.
    if history and history[-1].get("role") == "user":
        if (history[-1].get("content") or "").strip() == (req.question or "").strip():
            history = history[:-1]
    answer = gemini.ask(req.question, history)
    
    is_followup = answer.startswith("FOLLOWUP_NEEDED:")
    clean_answer = answer.replace("FOLLOWUP_NEEDED:", "").strip() if is_followup else answer
    
    return ChatResponse(answer=clean_answer, is_followup=is_followup)
