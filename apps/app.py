
# import packages
import os
from dotenv import load_dotenv
import openai
import streamlit as st


# Load environment variables from .env file
load_dotenv()

# local LLM (Ollama) config
base_url = os.getenv("OLLAMA_BASE_URL")
api_key = os.getenv("OLLAMA_API_KEY", "dummy")
model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")

client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key
)

@st.cache_data # cache the response to avoid repeated calls for same input
def get_response(user_prompt: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=200,   # ← din klient forventer max_tokens
    )
    return response.choices[0].message.content

st.title("Local LLM with Ollama and Streamlit")
st.write("Model:", model_name)

# add text for the user prompt
user_prompt = st.text_input(
    "Enter your prompt:", 
    "example: Explain generative AI in one sentence."
)

#add a slider for temperature
temperature = st.slider(
    "Model temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7, 
    step=0.01,
    help="Controls randomness: 0 = deterministic, 1 = more creative"
)


#if st.button("Generate Response"):
with st.spinner("AI is working..."): # show spinner while waiting for response
    reply = get_response(user_prompt, temperature) # generate response from the model
    st.write(reply) # display response

