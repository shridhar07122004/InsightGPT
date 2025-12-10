import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import nltk
import os

# 🔹 Set Groq API Token (replace with your own from console.groq.com)
os.environ["GROQ_API_KEY"] = "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"

# ✅ Download NLTK punkt tokenizer if not already present
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
with st.spinner("Downloading NLTK data..."):
    nltk.download('punkt', download_dir=nltk_data_dir)
    st.success("NLTK data downloaded successfully!")

# --- UI ---
st.title("InsightGPT 📈 ")
st.sidebar.title("Enter Your URLs Here")

# --- Session State ---
if "URLS_INPUT" not in st.session_state:
    st.session_state.URLS_INPUT = []
if "check" not in st.session_state:
    st.session_state.check = False
if "vectorindex" not in st.session_state:
    st.session_state.vectorindex = None

main_placeholder = st.empty()

# 🔹 Groq LLM (using LLaMA-3.1 8B Instant)
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# --- Sidebar input for URLs ---
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i+1}")
    if len(url) > 0:
        st.session_state.URLS_INPUT.append(url)

# --- Button to process URLs ---
process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=st.session_state.URLS_INPUT)
    main_placeholder.text("🔄 Loading data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("✂️ Splitting text...")
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    st.session_state.vectorindex = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorindex.save_local("faiss_index")

    st.session_state.check = True
    main_placeholder.success("✅ Processing Complete! You can now ask questions.")

# --- Query Section ---
if st.session_state.check:
    query = st.text_input("🔍 Ask your question:")
    if query:
        vectorindex = FAISS.load_local(
            "faiss_index",
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorindex.as_retriever()
        )

        response = chain({"question": query}, return_only_outputs=True)

        st.write("### ✅ Answer:", response['answer'])
       # st.write("### 📚 Sources:", response['sources'])
