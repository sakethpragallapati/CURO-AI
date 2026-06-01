# CURO AI

**Live Deployment:** [https://curo-ai-delta.vercel.app/](https://curo-ai-delta.vercel.app/)

[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28.svg?logo=firebase&logoColor=black)](https://firebase.google.com/)
<br>
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C.svg)](https://langchain.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-018bff.svg?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF4B4B.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-f55036.svg)](https://groq.com/)
<br>
[![Vercel](https://img.shields.io/badge/Vercel-000000?logo=vercel&logoColor=white)](https://vercel.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)

CURO AI is an advanced Clinical Retrieval-Augmented Generation (RAG) assistant engineered to support symptom analysis, interactive clinical triage, and secure health record management. By integrating state-of-the-art Large Language Models (LLMs) with graph databases and vector search, CURO AI delivers highly accurate, grounded medical insights and Differential Diagnosis (DDx) logic.

## Table of Contents
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Core Capabilities](#core-capabilities)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

---

## Architecture & Tech Stack

CURO AI employs a decoupled microservices architecture, separating the interactive client application from the computationally intensive RAG and NLP pipelines.

### Frontend Client
- **Framework:** Next.js 14 (App Router) & React 18
- **Styling:** Tailwind CSS for responsive, utility-first design
- **Visualization:** React Force Graph 2D for interactive knowledge graph rendering
- **Authentication:** Firebase Auth

### Backend Engine
- **Framework:** FastAPI (Python) for high-performance, asynchronous API endpoints
- **AI/NLP Pipeline:** LangChain integrated with Groq API for rapid LLM inference
- **Data Persistence:** 
  - **Neo4j:** Graph database for mapping clinical concepts and relationships
  - **ChromaDB:** Local vector store for document embeddings and semantic search
- **External Integrations:** OpenAlex (academic research retrieval), Exa (web search)
- **Audio Processing:** Server-side Whisper integration for high-fidelity Audio Speech Recognition (ASR)

---

## Core Capabilities

### Clinical Analysis & Triage
- **Grounded Symptom Analysis:** Leverages a sophisticated clinical RAG pipeline to process patient symptoms, outputting detailed DDx logic, a winning diagnosis, and retrieved medical abstracts.
- **Agent-Driven Triage:** An interactive chat module that dynamically generates relevant follow-up questions to comprehensively assess the user's condition.
- **Continuous Contextual Dialogue:** Enables ongoing chat sessions securely grounded in the initial clinical analysis context.

### Health Records Vault
- **Secure Document Processing:** Allows bulk uploading of medical records (PDFs). The system performs OCR, chunking, and embedding generation prior to storing the data in a persistent, user-isolated ChromaDB collection.
- **Semantic Querying:** Users can interrogate their uploaded health records using natural language, significantly accelerating data retrieval from dense medical files.

### Interactive Dashboard & Visualizations
- **Knowledge Graphs:** Dynamically visualizes the relationships between symptoms, diagnoses, and treatments using real-time Neo4j queries.
- **Fluid UI/UX:** Integrates modern React animation libraries to provide a responsive, professional, and accessible user interface.

---

## Getting Started

### Prerequisites
- **Node.js** (v18 or higher) and npm
- **Python** (v3.8 or higher)
- Active accounts/API keys for: **Firebase**, **Neo4j** (AuraDB or local), **Groq**, and **Exa**.

### Backend Setup

1. **Clone and navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Initialize a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   Copy the example environment file and populate it with your specific API credentials.
   ```bash
   cp .env.example .env
   ```

5. **Launch the FastAPI Server:**
   ```bash
   python main.py
   ```
   *The API will be available at `http://localhost:8000`. Swagger documentation can be accessed at `http://localhost:8000/docs`.*

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Environment Configuration:**
   Copy the example environment file and add your Firebase configuration details.
   ```bash
   cp .env.example .env
   ```

4. **Launch the Development Server:**
   ```bash
   npm run dev
   ```
   *The application will be available at `http://localhost:3000`.*

---

## Deployment

CURO AI uses a fully automated CI/CD pipeline managed by GitHub Actions (`.github/workflows/deploy-backend.yml`).

- **Frontend (Vercel):** The Next.js application is deployed to Vercel. The GitHub Action utilizes the Vercel CLI to securely pull configuration, build the production bundle, and deploy.
- **Backend (Hugging Face Spaces):** The FastAPI Python backend runs inside a Docker container on Hugging Face Spaces. The GitHub Action automatically force-pushes the `backend/` directory to the Space upon changes to the `master` branch.

All secrets (Firebase, Groq, Neo4j, etc.) are securely managed via Vercel Environment Variables and Hugging Face Secrets.

---

## Project Structure

```text
CURO-AI/
├── backend/                # FastAPI application and AI logic
│   ├── main.py             # Entry point for API routes
│   ├── curo_logic.py       # Core LangChain and RAG pipeline implementation
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Backend environment template
└── frontend/               # Next.js web application
    ├── app/                # Next.js App Router pages
    ├── components/         # Reusable React components
    ├── lib/                # Utility functions and Firebase initialization
    ├── package.json        # Node.js dependencies
    └── .env.example        # Frontend environment template
```
