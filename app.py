import os 
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveTextSplitter
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

st.tiitle("Gemma Model Document Q&A")

llm =  ChatGroq(groq_api_key=groq_api_key,model_name="Gemma_7b-it")

prompt = ChatPromptTemplate.from_template(""" Answer the questions based on the provided context only. Please provide the most accuatre  response on the questions below. <contest> {context} <context> Question: {input} """)

def vector_embedding():