import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq Demo")

if not os.path.exists("./refdata"):
    st.error("refdata directory not found!")
    st.stop()

pdf_files = [f for f in os.listdir("./refdata") if f.endswith('.pdf')]
if not pdf_files:
    st.error("No PDF files found in refdata directory!")
    st.stop()
else:
    st.success(f"There are {len(pdf_files)} PDF files available to refer")


@st.cache_resource
def initialize_vector_store():
    """Initialize and cache the vector store to avoid reloading on every run"""
    try:
        with st.spinner("Loading documents..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            loader = PyPDFDirectoryLoader("./refdata")
            docs = loader.load()
            
            st.write(f"✅ Loaded {len(docs)} documents")
            

            non_empty_docs = [doc for doc in docs if doc.page_content.strip()]
            #st.write(f"✅ Non-empty documents: {len(non_empty_docs)}")
            
            if not non_empty_docs:
                st.error("All documents appear to be empty. Check if PDFs contain extractable text.")
                st.stop()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            

            final_documents = text_splitter.split_documents(docs)
            
            #st.write(f"✅ Created {len(final_documents)} chunks")
            
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


if "vectors" not in st.session_state:
    st.session_state.vectors = initialize_vector_store()


@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question in atleast 100 words.
    Keep your answer concise and focused.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

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
                        st.write(f"**Document {i+1}:**")
                        st.write(doc.page_content)
                        st.write("--------------------------------")
                else:
                    st.write("No context found in response")
        else:
            st.error("No response generated. Check your query and try again.")
            
    except Exception as e:
        st.error(f"Error processing query: {e}")