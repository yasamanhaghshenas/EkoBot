---
title: EkoBot
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# EkoBot

AI-powered academic assistant for İzmir University of Economics.

## About

EkoBot is an intelligent chatbot that helps students of İzmir University of Economics with course information, curriculum planning, graduation requirements, and academic advising using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- 🔍 Course information lookup with evaluation systems
- 📋 Curriculum planning and course recommendations
- 🎓 Graduation requirements checking
- 📚 Comprehensive course database with syllabi
- 🧠 AI-powered responses using Llama 3.1 (via Groq API)
- 💾 Vector similarity search for relevant course information

## Technology Stack

- **Backend**: FastAPI + Uvicorn
- **Vector Database**: ChromaDB
- **LLM**: Groq API (Llama 3.1 8B Instant)
- **Embeddings**: Sentence Transformers
- **Data**: Course catalogs, curriculum data, IEU regulations

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Chat endpoint

### Chat Request Format

```json
{
  "question": "What is the grading system for MATH 153?",
  "history": []
}
```

### Chat Response Format

```json
{
  "answer": "MATH 153 - Calculus I\n\nEvaluation System:\n- Quizzes / Studio Critiques: 4 time(s), 20% weight\n- Midterm: 1 time(s), 30% weight\n- Final Exam: 1 time(s), 50% weight",
  "is_followup": false,
  "options": null
}
```

## Environment Variables

- `GROQ_API_KEY` - Your Groq API key for LLM access

## Deployment

This Space uses Docker for deployment. The application automatically:
1. Installs all Python dependencies from `requirements.txt`
2. Builds the ChromaDB vector database from JSON files
3. Starts the FastAPI server on port 7860

---

Built for İzmir University of Economics students.
