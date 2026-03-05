import streamlit as st
import os
import tempfile
import re
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Chat with PDF (GPT Style)",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 Chat with PDF (Offline GPT Style)")

# ----------------- SESSION STATE -----------------
if "docs_text" not in st.session_state:
    st.session_state.docs_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("📂 Upload Document(s)")
    files = st.file_uploader(
        "Upload PDF / TXT / PPTX (multiple allowed)", 
        type=["pdf","txt","pptx"], 
        accept_multiple_files=True
    )
    st.markdown("---")
    st.info("Offline mode: No API key needed ✅")
    if st.button("Clear History & Documents"):
        st.session_state.docs_text = ""
        st.session_state.chat_history = []

# ----------------- HELPER FUNCTIONS -----------------
def remove_page_numbers(text):
    return re.sub(r'\b\d{1,4}\b', '', text)

def preprocess_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def get_relevant_sentences(question, sentences, top_k=6):
    if not sentences:
        return []
    vectorizer = TfidfVectorizer().fit(sentences + [question])
    vectors = vectorizer.transform(sentences + [question])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = cosine_sim.argsort()[-top_k:][::-1]
    return [sentences[i] for i in top_indices]

def generate_gpt_style_answer(question, context):
    context_clean = remove_page_numbers(context)
    sentences = preprocess_text(context_clean)
    relevant_sentences = get_relevant_sentences(question, sentences)
    
    if not relevant_sentences:
        return "❌ No documents to generate answer."
    
    answer = ""
    for i, s in enumerate(relevant_sentences, 1):
        answer += f"**Point {i}:** {s}\n\n"
    return answer.strip()

# ----------------- LOAD DOCUMENTS -----------------
if files:
    with st.spinner("Processing files..."):
        all_text = []
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(f.getvalue())
                path = tmp.name
            try:
                if f.name.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif f.name.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                elif f.name.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(path)
                else:
                    continue
                docs = loader.load()
                for d in docs:
                    all_text.append(remove_page_numbers(d.page_content))
            finally:
                os.unlink(path)
        st.session_state.docs_text = "\n".join(all_text)
        st.success(f"✅ {len(files)} document(s) loaded successfully!")

# ----------------- CHAT INTERFACE -----------------
st.header("❓ Ask a Question")
question = st.text_input("Type your question here:")

if st.button("Send") and question:
    with st.spinner("Generating answer..."):
        context = st.session_state.docs_text
        answer = generate_gpt_style_answer(question, context) if context else "❌ No documents uploaded yet."
        st.session_state.chat_history.append({"question": question, "answer": answer})

# ----------------- DISPLAY CHAT HISTORY -----------------
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(
            f"<div style='background-color:#E0F7FA;padding:10px;border-radius:10px;margin-bottom:5px;'><b>You:</b> {chat['question']}</div>", 
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='background-color:#FFF3E0;padding:10px;border-radius:10px;margin-bottom:10px;'><b>🤖 Bot:</b><br>{chat['answer']}</div>", 
            unsafe_allow_html=True
        )

st.markdown("Built with ❤️ using Streamlit (Offline GPT Style)")



