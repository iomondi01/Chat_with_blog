import streamlit as st
import chromadb
import os
#from mistralai import Mistral
from dotenv import load_dotenv
from mistralai.client import MistralClient

load_dotenv()
#MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)
chroma_client = chromadb.PersistentClient(path="chroma_blog_db")
collection = chroma_client.get_or_create_collection("blog_rag_collection")

st.set_page_config(page_title="Chat with My Blog", layout="centered")
st.title("💬 Chat with My Blog")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def query_context(question, n=3):
    result = collection.query(query_texts=[question], n_results=n)
    return "\n\n".join(result["documents"][0]) if result["documents"] else ""

def generate_reply(question):
    context = query_context(question)
    messages = [
        {"role": "system", "content": f"You are a helpful assistant that answers based on this blog context:\n{context}"},
        {"role": "user", "content": question}
    ]
    response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return response.choices[0].message["content"]

query = st.text_input("Ask something about the blog:")
if query:
    answer = generate_reply(query)
    st.session_state.chat_history.append((query, answer))
    st.write("🤖", answer)

for q, a in st.session_state.chat_history:
    st.write(f"**You:** {q}")
    st.write(f"**Assistant:** {a}")

