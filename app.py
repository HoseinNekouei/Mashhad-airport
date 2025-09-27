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
import asyncio
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

    async def load_documents(self):
        documents = []
        
        for pdf in self.pdf_paths:
            pdf_path = (self.project_root / pdf).resolve()
        
            if not pdf_path.exists():
                st.error(f"File not found: {pdf_path}")
                continue
        
            try:
                loader = PyPDFLoader(str(pdf_path))

                pdf_docs = await asyncio.to_thread(loader.load)
                documents.extend(pdf_docs)

            except Exception as e:
                st.error(f"Error loading PDF {pdf_path}: {e}")
        
        return documents

    # ---------------- EMBEDDINGS ----------------
def load_embeddings():
    config= ConfigManager()
    embedding_model_name= config.get('embedding', 'model_name')  

    embedding_model = OllamaEmbeddings(
        model= 'aligh4699/heydariAI-persian-embeddings')
    
    return embedding_model


async def load_pdf_documents():
    config = ConfigManager()
    pdf_paths = config.get("data", "pdf_paths", [])
    
    if not pdf_paths:
        st.warning("No PDF files provided in config.")
        return []
    
    pdf_manager = PDFManager(pdf_paths)
    return await pdf_manager.load_documents()


def clean_duplicates(text):
    # Replace 2+ consecutive same letters with just one
    return re.sub(r'(.)\1+', r'\1', text)


def hash_text(text:str)-> str:
    hash_text =hashlib.sha256(text.encode()).hexdigest()

    return(hash_text)

def split_documents_into_chunks(documents: List[Any]) -> List[str]:

    config = ConfigManager()
    chunk_size = config.get("rag", "chunk_size", 1000)
    chunk_overlap = config.get("rag", "chunk_overlap", 200)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    texts= ""
    for doc in documents:
            texts += doc.page_content

    clean_text= clean_duplicates(texts)
    chunks = text_splitter.split_text(clean_text)
    
    return chunks


#---------------------VECTOR STORE ------------------------------
class VectorStoreCache:
    """ ChromaDB-based Vector Store cache"""

    def __init__(self, persist_dir='./Chroma_Store'):
        self.persist_dir = persist_dir
        self.embedding_model= load_embeddings()
        self.collection = None

    async def load_or_create(self, text_chunks: List[str]):

        # Load Existing ChromaDB or create a new with only missing chunks
        if os.path.exists(self.persist_dir) and os.listdir((self.persist_dir)):
            #load existing collection
            self.collection= Chroma(
                embedding_function= self.embedding_model,
                persist_directory= self.persist_dir
            )

        else:
            try:
            #create a new collection
                self.collection= await asyncio.to_thread(
                    Chroma.from_texts,
                    texts= text_chunks,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_dir
                    )
                                
                return self.collection
                     
            except Exception as e:
                st.error(f"Unexpected error while creating ChromaDB collection: {e}")
                st.stop()

        # check with chunks are missing
        existing_ids= set(self.collection.get()['ids'])

        # Filter out chunks that already exist
        new_chunks, new_ids= [],[]
        for chunk in text_chunks:
            chunk_id= hash_text(chunk)
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if new_chunks:
            await asyncio.to_thread(
                self.collection.add_texts,
                ids = new_ids,
                texts= new_chunks
                )


    async def similarity_search(self, query: str, k=4):

        if not self.collection:
            raise ValueError('Vecto store not loaded. call create_or_load() first')            

        return await asyncio.to_thread(self.collection.similarity_search, query, k=k)


# ---------------- RAG RESPONSE ----------------
async def get_response(query, chat_history):

    documents = await load_pdf_documents()

    if not documents:
        return "No documents available to provide context."

    text_chunks = split_documents_into_chunks(documents)
    vector_cache = VectorStoreCache()
    await vector_cache.load_or_create(text_chunks)

    try:
        result_docs = await vector_cache.similarity_search(query, k=4)
    
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        st.stop()

    if not result_docs:
        st.warning("No relevant content found for your query.")
        return

    context = "\n\n".join(result_doc.page_content for result_doc in result_docs)

    # Structured answer model prompt
    template = """
    You are a helpful AI assistant. Answer the user questions considering the context and the chat history.
    If you don't know the answer, just say you don't know. Don't try to make up:

    Chat history: {chat_history}
    User question: {question}
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(
        model='gemma3:4b',
        temperature=0.2
    )

    output_parser= StrOutputParser()

    chain = prompt | llm | output_parser

    try:
        response = await chain.ainvoke({
            "chat_history": chat_history,
            "question": query,
            "context": context,
        })

        return response

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating the response."



if __name__== '__main__':
    print(load_embeddings())


