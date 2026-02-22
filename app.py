import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------
# Load .env file
# ----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.warning("Google API key not found! Please add it in your .env file.")
    st.stop()

# ----------------------------
# PDF Text Extraction
# ----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ----------------------------
# Split text into chunks
# ----------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# ----------------------------
# Create FAISS vector store
# ----------------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# ----------------------------
# Setup conversational QA chain (Google Gemini)
# ----------------------------
def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    prompt_template = """
    Answer the question as thoroughly as possible using the context below. 
    If the answer is not present, generate the most reasonable answer based on the PDFs.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# ----------------------------
# Handle user input
# ----------------------------
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS vector store
    vector_store = st.session_state.get("vector_store", None)
    if vector_store is None:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload PDFs and process them first!")
            return
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state["vector_store"] = vector_store

    # Retrieve relevant chunks (semantic search)
    docs = vector_store.similarity_search(user_question, k=5)

    # Load QA chain
    chain = st.session_state.get("qa_chain", None)
    if chain is None:
        chain = get_conversational_chain()
        st.session_state["qa_chain"] = chain

    # Get answer
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Store conversation
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []
    st.session_state["conversation"].append((user_question, response["output_text"]))

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Chat PDF with Google Gemini")
    st.header("Chat with PDF 💁")

    with st.sidebar:
        st.title("Upload PDF Files")
        pdf_docs = st.file_uploader("Upload PDFs and click Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state["vector_store"] = vector_store
                    st.success("PDFs processed! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # User input
    user_question = st.text_input("Ask a question from your PDF files:")
    if user_question:
        user_input(user_question)

    # Display conversation
    if "conversation" in st.session_state:
        for i, (q, a) in enumerate(st.session_state["conversation"]):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
            st.markdown("---")

if __name__ == "__main__":
    main()



