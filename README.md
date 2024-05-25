# Gemma Model Document Q&A

This project demonstrates a Streamlit application for querying documents using the Gemma Model, powered by ChatGroq's Llama3-8b-8192 model and Google Generative AI embeddings for vector-based retrieval. The application allows users to load PDF documents, generate embeddings, and query the documents to receive accurate responses based on the provided context.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gemma-model-qa.git
    cd gemma-model-qa
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory with your Groq and Google API keys:
    ```env
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

4. Ensure you have your PDF documents in a directory named `Files` in the root directory.

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Streamlit Application Code

```python
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # Vector store DB
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Vector embedding technique
from dotenv import load_dotenv

load_dotenv()

# Load the Groq and Google API keys from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only. Please provide the most accurate response to the questions below. <context> {context} <context> Question: {input}
""")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./Files")  # Data ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document loadings
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Text splitter
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting the documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector store

prompt1 = st.text_input("What do you want to ask from the document?")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------")
