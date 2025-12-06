# GenAI Prototype Workspace

This repository contains a set of experimental, modular Streamlit applications that explore local LLM workflows (via Ollama) and data-driven analysis pipelines. The project currently includes two main directions: a PDF reader powered by local LLM question-answering, and a set of data-analysis apps that combine classic analytics with GenAI-driven insights.

## 1️⃣ PDF Reader & Local LLM QA (`app_reader01.py`)

An interactive Streamlit app that allows users to upload a PDF, extract text, and ask questions grounded strictly in the document content. The app communicates with a local LLM (Llama 3.1, Qwen, Mistral, etc. running through Ollama) and includes caching, improved grounding prompts, and robust PDF processing. This serves as a prototype for document understanding, lightweight RAG workflows, and future multi-document reasoning.

## 2️⃣ GenAI + Data Analysis Apps

Additional Streamlit apps explore ingestion and analysis of structured datasets, including sentiment scoring, clustering, visual analytics (Plotly, Altair), and LLM-assisted reporting. These tools function as a sandbox for experimenting with GenAI copilots on real-world data, using both local and remote LLM endpoints.

## 🚀 Running the Apps

### 1. Create and activate the virtual environment
python -m venv .venv

.venv\Scripts\Activate.ps1

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run streamlit app 
streamlit run apps/app_reader01.py
# or
streamlit run app_reader.py

