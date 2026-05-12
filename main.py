from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import gemini
import os

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
    history: Optional[List[Message]] = None

class ChatOption(BaseModel):
    label: str
    value: str


class ChatResponse(BaseModel):
    answer: str
    is_followup: bool = False
    options: Optional[List[ChatOption]] = None

@app.get("/")
def root():
    # Serve the website index.html if it exists, otherwise return API status
    website_path = os.path.join(os.path.dirname(__file__), "website", "index.html")
    if os.path.exists(website_path):
        return FileResponse(website_path, media_type="text/html")
    return {"message": "EkoBot is running!"}

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    history = [{"role": m.role, "content": m.content} for m in (req.history or [])]

    # Some clients include the current question in the history payload.
    # The backend already receives it separately as `question`.
    if history and history[-1].get("role") == "user":
        if (history[-1].get("content") or "").strip() == (req.question or "").strip():
            history = history[:-1]
    client_type = (request.headers.get("X-Client-Type") or "").strip().lower()
    result = gemini.ask_with_options(req.question, history, client_type=client_type)

    return ChatResponse(
        answer=result.get("answer", ""),
        is_followup=bool(result.get("is_followup", False)),
        options=result.get("options"),
    )
