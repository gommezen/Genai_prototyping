import os
import streamlit as st
import openai
from dotenv import load_dotenv
from pypdf import PdfReader

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

base_url = os.getenv("OLLAMA_BASE_URL")
api_key = os.getenv("OLLAMA_API_KEY", "dummy")
model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")

client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key
)

st.title("📄 Chat with your local PDFs")

st.write("Upload a PDF and ask questions about its content.")

# ----------------------------
# PDF upload section
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    # save to session_state
    st.session_state["pdf_text"] = text

    st.success("PDF uploaded and processed!")

    with st.expander("Show extracted text"):
        st.write(text[:3000] + "...\n\n[Text truncated]")


# ----------------------------
# Ask a question about the PDF
# ----------------------------
question = st.text_input("Ask a question about the PDF:")

def ask_model(question, document_text):
    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the document below.

DOCUMENT:
\"\"\"
{document_text}
\"\"\"

QUESTION:
{question}

If the answer is not in the document, say: 'The document does not contain enough information.'
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )

    return response.choices[0].message.content


if st.button("Ask the PDF"):
    if "pdf_text" not in st.session_state:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_model(question, st.session_state["pdf_text"])
            st.write("### Answer:")
            st.write(answer)
