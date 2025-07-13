import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

# CRITICAL export before running the code
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq Demo")

# Sidebar for file upload
st.sidebar.header("📁 Document Management")
st.sidebar.markdown("Upload PDF files to add to the knowledge base")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF files to add to the knowledge base"
)

# Option to use existing refdata folder
use_refdata = st.sidebar.checkbox("Also use files from refdata folder", value=True)

def load_documents_from_uploads(uploaded_files):
    """Load documents from uploaded files"""
    docs = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Load the PDF
                loader = PyPDFLoader(tmp_file_path)
                file_docs = loader.load()
                
                # Add filename to metadata
                for doc in file_docs:
                    doc.metadata['filename'] = uploaded_file.name
                
                docs.extend(file_docs)
                
            except Exception as e:
                st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
    
    return docs

def load_documents_from_refdata():
    """Load documents from refdata folder"""
    docs = []
    
    if os.path.exists("./refdata"):
        pdf_files = [f for f in os.listdir("./refdata") if f.endswith('.pdf')]
        if pdf_files:
            try:
                loader = PyPDFDirectoryLoader("./refdata")
                docs = loader.load()
                
                # Add source folder to metadata
                for doc in docs:
                    doc.metadata['source_folder'] = 'refdata'
                    
            except Exception as e:
                st.sidebar.error(f"Error loading refdata folder: {e}")
    
    return docs

# Initialize vector store only once
def initialize_vector_store(uploaded_files, use_refdata):
    """Initialize and cache the vector store to avoid reloading on every run"""
    try:
        with st.spinner("Loading documents..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Load documents from different sources
            all_docs = []
            
            # Load from uploaded files
            if uploaded_files:
                uploaded_docs = load_documents_from_uploads(uploaded_files)
                all_docs.extend(uploaded_docs)
                st.write(f"✅ Loaded {len(uploaded_docs)} documents from uploaded files")
            
            # Load from refdata folder if enabled
            if use_refdata:
                refdata_docs = load_documents_from_refdata()
                all_docs.extend(refdata_docs)
                st.write(f"✅ Loaded {len(refdata_docs)} documents from refdata folder")
            
            if not all_docs:
                st.error("No documents loaded. Please upload PDF files or enable refdata folder.")
                st.stop()
            
            st.write(f"✅ Total documents loaded: {len(all_docs)}")
            
            # Debug: Check content of first few documents
            non_empty_docs = [doc for doc in all_docs if doc.page_content.strip()]
            st.write(f"✅ Non-empty documents: {len(non_empty_docs)}")
            
            if not non_empty_docs:
                st.error("All documents appear to be empty. Check if PDFs contain extractable text.")
                st.stop()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            
            # Remove the [:50] limit to process all documents
            final_documents = text_splitter.split_documents(all_docs)
            
            st.write(f"✅ Created {len(final_documents)} chunks")
            
            if not final_documents:
                st.error("No document chunks created. Check if PDFs contain valid text.")
                st.stop()
        
        with st.spinner("Creating vector store..."):
            vectors = FAISS.from_documents(final_documents, embeddings)
        
        st.success("✅ Vector store created successfully!")
        return vectors
        
    except Exception as e:
        st.error(f"Error during initialization: {e}")
        st.stop()

# Create a cache key based on uploaded files and settings
file_names = [f.name for f in uploaded_files] if uploaded_files else []
cache_key = f"vectors_{hash(tuple(file_names))}_{use_refdata}"

# Get the cached vector store or create new one
if cache_key not in st.session_state:
    if uploaded_files or use_refdata:
        st.session_state[cache_key] = initialize_vector_store(uploaded_files, use_refdata)
        st.session_state['current_cache_key'] = cache_key
    else:
        st.warning("Please upload PDF files or enable the refdata folder to proceed.")
        st.stop()

vectors = st.session_state[cache_key]

# Initialize LLM and chains (cached)
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question in about 100 words.
    Keep your answer concise and focused.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Display current document status
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Status:**")
if uploaded_files:
    st.sidebar.success(f"📄 {len(uploaded_files)} uploaded files")
    for file in uploaded_files:
        st.sidebar.text(f"• {file.name}")

if use_refdata and os.path.exists("./refdata"):
    refdata_files = [f for f in os.listdir("./refdata") if f.endswith('.pdf')]
    if refdata_files:
        st.sidebar.success(f"📁 {len(refdata_files)} refdata files")
        for file in refdata_files[:5]:  # Show first 5
            st.sidebar.text(f"• {file}")
        if len(refdata_files) > 5:
            st.sidebar.text(f"... and {len(refdata_files) - 5} more")

# Reset button
if st.sidebar.button("🔄 Reset and Reload Documents"):
    # Clear all vector caches
    for key in list(st.session_state.keys()):
        if key.startswith("vectors_"):
            del st.session_state[key]
    
    # Clear Streamlit cache to force rebuilding
    st.cache_resource.clear()
    
    st.sidebar.success("Cache cleared! Rebuilding embeddings...")
    st.rerun()

# User input
prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    try:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt_input})
        print("Response time:", time.process_time() - start)
        
        if response and 'answer' in response:
            st.write(response['answer'])
            
            with st.expander("Document Similarity Search"):
                if 'context' in response:
                    for i, doc in enumerate(response["context"]):
                        # Show filename if available
                        filename = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
                        st.write(f"**Document {i+1}:** {filename}")
                        st.write(doc.page_content)
                        st.write("--------------------------------")
                else:
                    st.write("No context found in response")
        else:
            st.error("No response generated. Check your query and try again.")
            
    except Exception as e:
        st.error(f"Error processing query: {e}")