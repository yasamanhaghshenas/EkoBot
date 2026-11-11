# EkoBot
### A Retrieval-Augmented Academic Assistant for İzmir University of Economics

EkoBot is a student-built academic information assistant designed to help İzmir University of Economics (IEU) students navigate course information and selected university resources. It combines a Python-based data collection pipeline, a ChromaDB vector store, and a conversational language model accessed through OpenRouter.

> **Project status:** Academic prototype. EkoBot is an independent student project and is not affiliated with, endorsed by, or operated by İzmir University of Economics. Its responses may be incomplete or outdated; students should verify important academic decisions against current official university sources.

---

## Overview

University information is often distributed across course pages, curriculum documents, and different service websites. EkoBot brings selected information from these public sources into a searchable knowledge base and uses retrieved context to support conversational answers.

The project focuses on:

- **Course information:** descriptions, prerequisites, credits, ECTS, weekly topics, and assessment information available on public course pages.
- **Academic information:** selected curriculum, department, academic-regulation, and university-service content collected from public IEU webpages.
- **Conversational interaction:** English- and Turkish-language questions, recent conversation context, and follow-up questions when more details are needed.
- **Source-aware retrieval:** semantic search over indexed documents to provide relevant context to the language model.

The system is intended as an information-navigation aid, not as an official academic advising or decision-making service.

## How it works

The data and response pipeline has six main stages:

```mermaid
flowchart TD
    A[Public IEU webpages] --> B[Python collection and parsing]
    B --> C[Cleaned and structured JSON]
    C --> D[Sentence Transformers embeddings]
    D --> E[ChromaDB vector index]
    F[Student question and recent chat history] --> G[FastAPI]
    G --> E
    E --> H[Retrieved context]
    H --> I[OpenRouter chat model]
    F --> I
    I --> J[Answer and follow-up metadata]
    J --> K[Web chat interface]
```

1. **Collect** — Python scripts retrieve information from selected publicly accessible IEU webpages.
2. **Parse and structure** — HTML is cleaned and relevant information is extracted into structured JSON records and text chunks.
3. **Embed** — Sentence Transformers converts text into vector representations using `all-MiniLM-L6-v2`.
4. **Index and retrieve** — ChromaDB stores the vectors and returns semantically relevant documents for a question.
5. **Generate** — the retrieved context and recent conversation history are sent to a chat model through the OpenRouter API.
6. **Respond** — the FastAPI backend returns the answer and, where applicable, follow-up information for the client.

## Data collection and provenance

The project team assembled its source list collaboratively and collected information page by page from publicly accessible pages on IEU's official website. The dataset was built by the project team; it was **not** supplied by the university, and no private university database was used.

The Python collection and processing work covers selected public information such as:

- Course descriptions, prerequisites, credits, ECTS, weekly topics, and assessment breakdowns.
- Department and curriculum information.
- Selected academic regulations and student-affairs information.
- Academic calendar, IT-support, career, and internship resources.

The collection scripts use source URLs, and source metadata is retained in the generated records where available so that information can be traced back to its originating webpage.

Examples of source areas include:

- [İzmir University of Economics](https://www.ieu.edu.tr/)
- [IEU Student Affairs](https://oim.ieu.edu.tr/)
- [IEU IT Support](https://itsupport.ieu.edu.tr/)
- [IEU Career](https://kariyer.ieu.edu.tr/)
- [IEU Computer Engineering](https://comp.izmirekonomi.edu.tr/)

Public accessibility does not necessarily grant permission to republish or commercially reuse website content. The generated data files are project snapshots; consult the relevant source websites before reusing their contents. The information may change, so the index should be refreshed when the source pages are updated.

## Technology stack

| Component | Technology |
|---|---|
| Backend API | Python, FastAPI, Uvicorn |
| Vector database | ChromaDB |
| Text embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Language-model access | OpenAI Python SDK, OpenRouter |
| Chat model configured in the current code | `meta-llama/llama-3-8b-instruct:free` |
| Web scraping and parsing | Requests, Beautiful Soup |
| Data format | JSON |
| Validation | Pydantic |
| Containerization | Docker |

## My contribution

I took primary responsibility for the Python web-scraping and data-extraction pipeline: designing and implementing the core scripts that collect public course information, parse webpage content, and transform the extracted material into structured JSON data.

I also contributed to the project's data preparation and indexing workflow, ChromaDB-based retrieval components, and the conversational backend. The work was collaborative: source URLs were compiled with a teammate, and development of shared project modules included contributions from both team members.

This project gave me practical experience with web data acquisition, information extraction, data structuring, vector search, API development, and integrating retrieval with a conversational language model.

## Project structure

| File or directory | Purpose |
|---|---|
| `main.py` | FastAPI application and chat endpoint |
| `gemini.py` | OpenRouter client, prompt construction, context retrieval, and conversation flow |
| `scraper.py` | Course and selected university-page collection |
| `scraper_curriculum.py`, `scraper_info.py` | Additional curriculum and university-information collection |
| `catalog.py`, `planning.py` | Course, department, curriculum, and planning-related data helpers |
| `embeddings.py` | Text embedding wrapper |
| `indexer.py` | Builds the ChromaDB knowledge index |
| `retrieval.py` | Course-focused retrieval helper functions |
| `website/` | Browser-based chat interface |
| `tests/`, `eval_questions.json` | Test and evaluation inputs |

## Run locally

### Requirements

- Python 3.11
- An OpenRouter API key
- Internet access for model and embedding downloads and for any source collection that requires live webpages

### 1. Clone the repository

```bash
git clone https://github.com/yasamanhaghshenas/EkoBot.git
cd EkoBot
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Configure the API key

Set your OpenRouter API key as an environment variable.

macOS / Linux:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key"
```

Never commit API keys, tokens, or other credentials.

### 4. Build the vector index

```bash
python indexer.py
```

### 5. Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open [http://localhost:7860](http://localhost:7860) for the web interface. FastAPI's interactive API documentation is available at [http://localhost:7860/docs](http://localhost:7860/docs).

### Docker

```bash
docker build -t ekobot .
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" ekobot
```

## API

### `GET /`

Serves the web interface when `website/index.html` is available; otherwise returns a basic API status response.

### `POST /chat`

Example request:

```json
{
  "question": "What are the prerequisites for SE 302?",
  "history": []
}
```

The response includes:

- `answer` — generated response text.
- `is_followup` — whether the response is part of a follow-up flow.
- `options` — optional suggested replies.
- `followup_mode` — optional follow-up handling mode.
- `session_language` — detected session language.

## Limitations and responsible use

- EkoBot is a prototype and has not been presented as an authoritative source of academic policy.
- Retrieval quality depends on the coverage, freshness, and structure of the collected webpages.
- Language-model responses can be incorrect or incomplete, even when relevant context is retrieved.
- The current repository does not publish a measured end-to-end answer-accuracy score.
- Always confirm prerequisites, graduation requirements, course availability, and other consequential decisions with IEU's current official sources.

## License

No open-source license has been added. Until a license is selected, the repository should be treated as **all rights reserved**; public visibility alone does not grant permission to reuse, modify, or redistribute the code or collected data.
