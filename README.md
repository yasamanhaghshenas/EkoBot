---
title: EkoBot
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# EkoBot

**An AI-powered academic assistant for İzmir University of Economics (IEU).**

EkoBot helps students find information about courses, curricula, academic regulations, and university services. It combines a retrieval pipeline built around IEU-specific source material with a conversational language model.

> EkoBot is an independent academic project. It is not an official university service, and its responses should be checked against the university's current regulations and official announcements.

## What it does

- Retrieves course information such as descriptions, prerequisites, instructors, credits, ECTS, and assessment breakdowns.
- Supports curriculum and semester-planning questions using department, study-year, semester, and curriculum-year context.
- Handles questions about graduation requirements, academic rules, and selected student services.
- Supports English and Turkish conversations.
- Maintains recent conversation history and can ask follow-up questions when more student context is needed.

## How it works

1. **Knowledge preparation** — course and university information is collected from structured JSON files and selected IEU web sources.
2. **Embeddings** — text is encoded with Sentence Transformers (all-MiniLM-L6-v2).
3. **Vector retrieval** — ChromaDB stores the indexed knowledge and retrieves relevant context for a question.
4. **Answer generation** — the retrieved context and recent conversation are passed to an OpenAI-compatible chat API hosted by OpenRouter.
5. **Response handling** — the API returns the answer together with follow-up metadata when additional information is required.

For course-code questions, the backend first attempts an exact lookup in the course data and then uses ChromaDB retrieval as a fallback. General questions use semantic retrieval to assemble context for the language model.

## Technology

| Layer | Technologies |
|---|---|
| API | Python, FastAPI, Uvicorn |
| Retrieval | ChromaDB |
| Embeddings | Sentence Transformers, all-MiniLM-L6-v2 |
| Language model access | OpenAI Python SDK, OpenRouter API |
| Data collection and parsing | Requests, Beautiful Soup |
| Validation | Pydantic |
| Deployment | Docker |

If the Sentence Transformers model cannot be loaded, the embedding wrapper includes a lightweight hash-based fallback. This fallback is intended for graceful degradation and is not equivalent in semantic quality to the transformer model.

## API

The FastAPI application exposes:

- GET / — serves the web interface when website/index.html is available; otherwise returns a basic status response.
- POST /chat — accepts a question and optional conversation history.

Example request:

~~~json
{
  "question": "What are the prerequisites for SE 302?",
  "history": []
}
~~~

The response includes an answer, an is_followup flag, optional options, an optional followup_mode, and the detected session_language.

Interactive API documentation is available at /docs when the application is running.

## Run locally

### 1. Clone the repository

~~~bash
git clone https://github.com/yasamanhaghshenas/EkoBot.git
cd EkoBot
~~~

### 2. Create an environment and install dependencies

Python 3.11 is used by the Docker image.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

On Windows PowerShell, activate the environment with:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

### 3. Configure the language-model API key

Set an OpenRouter API key in your environment:

~~~bash
export OPENROUTER_API_KEY="your_api_key"
~~~

Do not commit API keys or other credentials to the repository. Keep them in environment variables or a local .env file that is excluded from version control.

### 4. Build the vector index and start the API

The startup script attempts to build the ChromaDB index before launching the server:

~~~bash
python indexer.py
uvicorn main:app --host 0.0.0.0 --port 7860
~~~

Alternatively, run the Docker image:

~~~bash
docker build -t ekobot .
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" ekobot
~~~

The indexer expects the relevant course, curriculum, and university-source data files to be present in the project directory.

## Configuration

| Setting | Purpose |
|---|---|
| OPENROUTER_API_KEY | Authenticates requests to OpenRouter |
| Model | meta-llama/llama-3-8b-instruct:free |
| ChromaDB collection | ieu_knowledge_base |
| General retrieval | Up to 5 context documents per question |

The model and retrieval settings are currently defined in gemini.py.

## Project structure

- main.py — FastAPI application and request/response models.
- gemini.py — language-model client, prompt construction, retrieval orchestration, and follow-up handling.
- retrieval.py — course-code retrieval, focused course responses, and retrieval fallback logic.
- embeddings.py — Sentence Transformers embedding wrapper and hash-based fallback.
- indexer.py — builds the ChromaDB knowledge index.
- scraper.py — collects course and selected university web-page content.
- catalog.py — course, department, curriculum, and source-catalog helpers.
- website/ — web client.
- tests/ and evaluation JSON files — test and evaluation inputs.

## Data and responsible use

The application uses course and university information represented in local data files and source catalogs. Before redistributing those files, verify that their contents may be shared publicly and that they do not contain non-public university material or personal information. Where possible, prefer a small, clearly attributed sample dataset and document how the full index can be generated.

EkoBot is a student project and an informational aid. University policies and course details can change; users should confirm consequential academic decisions with IEU's official sources.

## License

No open-source license has been added yet. Until a license is selected, reuse and redistribution permissions are not granted by this repository.
