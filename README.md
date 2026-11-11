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

EkoBot helps students find information about courses, curricula, academic regulations, and university services. It combines a retrieval pipeline built around publicly available IEU web content with a conversational language model.

> EkoBot is an independent student project, not an official university service. Its answers are informational and should be checked against the university's current official pages.

## What it does

- Retrieves course information such as descriptions, prerequisites, instructors, credits, ECTS, and assessment breakdowns.
- Supports curriculum and semester-planning questions using department, study-year, semester, and curriculum-year context.
- Handles questions about graduation requirements, academic rules, and selected student services.
- Supports English and Turkish conversations.
- Maintains recent conversation history and can ask follow-up questions when more student context is needed.

## Data collection and preparation

The project team built its own dataset by collecting information page by page from publicly accessible pages on IEU's official website. No private university database or university-provided internal dataset was used.

The Python collection and processing scripts:

- collect course syllabus information from public course pages;
- extract course descriptions, prerequisites, credits, ECTS, weekly topics, and assessment information;
- collect publicly available curriculum and department information;
- scrape selected student-affairs, academic-calendar, IT-support, career, and internship pages;
- clean the extracted HTML, organize the content, and store the resulting records in JSON files for indexing.

Each scraped course record retains its source URL, and web-page chunks include source URL metadata. This makes the origin of retrieved information traceable to the public page from which it was collected.

Examples of source areas used by the scraper include:

- IEU official website: https://www.ieu.edu.tr/
- IEU Student Affairs: https://oim.ieu.edu.tr/
- IEU IT Support: https://itsupport.ieu.edu.tr/
- IEU Career: https://kariyer.ieu.edu.tr/
- IEU Computer Engineering pages: https://comp.izmirekonomi.edu.tr/

The collected JSON files are generated project data, not files supplied by the university. Since official pages may change, the dataset and vector index should be refreshed periodically.

## How it works

1. **Collection** — Python scripts retrieve information from selected public IEU pages.
2. **Preparation** — extracted text is cleaned, structured, and stored in JSON files.
3. **Embeddings** — text is encoded with Sentence Transformers (all-MiniLM-L6-v2).
4. **Vector retrieval** — ChromaDB stores the indexed knowledge and retrieves relevant context for a question.
5. **Answer generation** — retrieved context and recent conversation are passed to an OpenAI-compatible chat API hosted by OpenRouter.
6. **Response handling** — the API returns the answer together with follow-up metadata when additional student context is required.

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

~~~bash
python indexer.py
uvicorn main:app --host 0.0.0.0 --port 7860
~~~

Alternatively, run the Docker image:

~~~bash
docker build -t ekobot .
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" ekobot
~~~

The indexer reads the available course, curriculum, department, and university-source JSON files and builds the ChromaDB collection.

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
- JSON data files — collected and structured project data used by the application.
- tests/ and evaluation JSON files — test and evaluation inputs.

## Responsible use and data freshness

EkoBot is a student-built informational tool. Course details, academic regulations, and university services can change. Users should verify consequential academic decisions against IEU's current official sources.

## License

No open-source license has been added yet. Until a license is selected, reuse and redistribution permissions are not granted by this repository.
