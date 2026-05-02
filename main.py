from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import asyncio
import gemini
from startup import build_db_if_empty


async def run_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, build_db_if_empty)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # run indexing in background so server starts immediately
    asyncio.create_task(run_startup())
    yield


app = FastAPI(title="EkoBot API", lifespan=lifespan)

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
    return {"message": "EkoBot is running!"}

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    try:
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
    except Exception as e:
        import traceback
        print(f"CHAT ERROR: {e}")
        print(traceback.format_exc())
        return ChatResponse(
            answer=f"Server error: {str(e)}", 
            is_followup=False
        )
