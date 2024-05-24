import os 
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vector sotre db
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Vector embedding technique

from dotenv import load_dotenv

load_dotenv()

##load the Groq and google api key for the .env file

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm =  ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(""" Answer the questions based on the provided context only. Please provide the most accuatre  response on the questions below. <contest> {context} <context> Question: {input} """)

def vector_embedding(): 
    
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model ="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./Files") #data ingestion step
        st.session_state.docs=st.session_state.loader.load() #document loadings
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #text splitter
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting the documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #vector store
        
prompt1=st.text_input("What you want to ask form the document?")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Stor DB is Ready")
    
import time 

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()  
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])
    
    #with a streamlit expander
    with st.expander("Document Similartiy Search"):
        #find the relevant chunks
        for i , doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------")      
        