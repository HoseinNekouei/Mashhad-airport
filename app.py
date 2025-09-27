# if your cloud using chromadb <= 35 using it othervise comment ignore it
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ------------------------------------------------------

import os
import re
import yaml
import pdb
import hashlib
from typing import List, Any
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings, ChatOllama


# ---------------- CONFIG MANAGER ----------------
class ConfigManager:
    """Handles loading of yaml configuration."""

    def __init__(self, path='config.yaml') -> None:
        self.path = path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.path, 'r') as file:
                return yaml.safe_load(file)
        
        except Exception as e:
            st.error(f'Error loading configuration {e}')
            return {}

    def get(self, section, key=None, default=None):
        if key:
            return self.config.get(section, {}).get(key, default)
        
        return self.config.get(section, default)
    

# ---------------- PDF MANAGER ----------------
class PDFManager:
    """Handles loading and processing of PDF files."""

    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
        self.project_root = Path(__file__).parent

    def load_documents(self):
        documents = []
        
        for pdf in self.pdf_paths:
            pdf_path = (self.project_root / pdf).resolve()
        
            if not pdf_path.exists():
                st.error(f"File not found: {pdf_path}")
                continue
        
            try:
                loader = PyPDFLoader(str(pdf_path))
                pdf_docs = loader.load()
                documents.extend(pdf_docs)

            except Exception as e:
                st.error(f"Error loading PDF {pdf_path}: {e}")
        
        return documents

    # ---------------- EMBEDDINGS ----------------
def load_embeddings():
    config= ConfigManager()
    embedding_model_name= config.get('embedding', 'model_name')  

    embedding_model = OllamaEmbeddings(
        model=embedding_model_name)
    
    return embedding_model


def load_pdf_documents():
    config = ConfigManager()
    pdf_paths = config.get("data", "pdf_paths", [])
    
    if not pdf_paths:
        st.warning("No PDF files provided in config.")
        return []
    
    pdf_manager = PDFManager(pdf_paths)
    return pdf_manager.load_documents()

def clean_duplicates(text):
    # Replace 2+ consecutive same letters with just one
    return re.sub(r'(.)\1+', r'\1', text)



if __name__== '__main__':
    print(load_embeddings())


