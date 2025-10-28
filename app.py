import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader

# ---------- Helper functions ----------

@st.cache_data
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qa_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_answer(question, embedding_model, qa_model):
    # Load FAISS and metadata
    index = faiss.read_index("faiss.index")
    with open("meta.pkl", "rb") as f:
        docs = pickle.load(f)

    # Search for most relevant chunks
    q_emb = embedding_model.encode([question], convert_to_numpy=True)
    _, idx = index.search(q_emb, 3)
    context = " ".join([docs[i] for i in idx[0]])

    # Prompt template
    prompt = f"""
You are a helpful and knowledgeable tutor specialized in this course.
Use the context below to answer the question accurately and clearly.

Context:
{context}

Question: {question}
Answer:
"""
    result = qa_model(prompt, max_length=250, do_sample=False)
    return result[0]["generated_text"]

# ---------- Streamlit Interface ----------

st.title("🎓 Course Q&A Chatbot (RAG System)")

pdf_file = st.file_uploader("📄 Upload your course PDF", type=["pdf"])

if pdf_file:
    text = load_pdf(pdf_file)
    chunks = chunk_text(text)
    st.success(f"PDF loaded — {len(chunks)} chunks created.")

    # Build FAISS index
    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open("meta.pkl", "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, "faiss.index")

    st.info("Index built. You can now ask your questions!")

    qa_model = load_qa_model()

    # Chat area
    question = st.text_input("💬 Ask a question about the course:")
    if question:
        with st.spinner("Thinking..."):
            answer = get_answer(question, embedding_model, qa_model)
        st.markdown(f"**Answer:** {answer}")
