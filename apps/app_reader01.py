import os
from typing import Union
from io import BytesIO

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from dotenv import load_dotenv
import openai
from pypdf import PdfReader


# ======================================================
# 1. Load environment variables for Ollama client
# ======================================================
load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL")
API_KEY = os.getenv("OLLAMA_API_KEY", "dummy")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")

# OpenAI client configured to talk to local Ollama server
client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ======================================================
# 2. Utility: PDF extraction (cached)
# ======================================================

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file: Union[UploadedFile, BytesIO]) -> str:
    """
    Extract plain text from a PDF uploaded through Streamlit.
    Streamlit's UploadedFile behaves like a BytesIO object, and PdfReader
    supports file-like objects, so no manual .read() needed.

    Returns:
        str: Extracted text (may be empty string if unreadable)
    """

    try:
        reader = PdfReader(file)
        text_parts = []

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)

        return "\n".join(text_parts)

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")


# ======================================================
# 3. Utility: Ask the LLM a question (cached)
# ======================================================

@st.cache_data(show_spinner=False)
def ask_llm(document_text: str, question: str, model_name: str, temperature: float = 0.1) -> str:
    """
    Ask a question about a given document using a local LLM (Ollama).

    Always returns a string—never None—so Streamlit rendering is safe.
    Cached so identical doc/question combinations do not repeat LLM inference.

    Returns:
        str: The model's answer or an error message.
    """

    if not document_text.strip():
        return "Error: No readable text could be extracted from the PDF."

    prompt = f"""
You are an assistant whose answers must be based *strictly* on the document content below.
If a question cannot be answered from the document, reply exactly with:

"The document does not contain enough information to answer this."

---------------- DOCUMENT ----------------
{document_text}
------------------------------------------

QUESTION:
{question}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300
        )
        return str(response.choices[0].message.content)

    except Exception as e:
        return f"LLM request failed: {e}"


# ======================================================
# 4. Streamlit UI
# ======================================================

st.title("📄 Local PDF Question-Answering (Ollama + Streamlit)")
st.caption("Upload a PDF and ask questions grounded strictly in its content.")

st.write("### 1️⃣ Upload a PDF File")
uploaded_pdf = st.file_uploader("Choose a PDF to analyze:", type=["pdf"])

document_text: str = ""

if uploaded_pdf is not None:
    try:
        document_text = extract_text_from_pdf(uploaded_pdf)
        st.success("PDF successfully uploaded and processed!")

        # Optional preview
        with st.expander("Show extracted text preview"):
            preview = document_text[:3000]
            if len(document_text) > 3000:
                preview += "\n\n... [Text truncated]"
            st.text(preview)

    except RuntimeError as e:
        st.error(str(e))
        document_text = ""


# Question input
st.write("### 2️⃣ Ask a question about the uploaded PDF")
question = st.text_input("Type your question:")

temperature = st.slider(
    "Model temperature (creativity vs precision):",
    0.0, 1.0, 0.1,
    help="0.0 = strict & factual, 1.0 = creative"
)

# Button logic
if st.button("Ask the PDF"):
    if uploaded_pdf is None:
        st.error("Please upload a PDF first.")
    elif not question.strip():
        st.error("Please enter a valid question.")
    else:
        with st.spinner("Analyzing document with local LLM..."):
            answer = ask_llm(document_text, question, MODEL_NAME, temperature)

        st.write("### 🧠 Answer")
        st.write(answer)
