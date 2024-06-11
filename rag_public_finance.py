# -*- coding: utf-8 -*-
# Retrieval-Augmented Generation (RAG) Chatbot for Finance

"""Access to files"""

import os
import markdown
path = os.path.join("data", "documents")
persist_dir = os.path.join("data", "storage")


"""Large Language Model"""

import torch
from llama_index.llms.groq import Groq

# Select a model from Groq API: "llama3-70b-8192" "llama3-8b-8192" "mixtral-8x7b-32768" "Gemma-7b-it"
model = "llama3-8b-8192"
llm = Groq(model=model, api_key=os.environ.get("GROQ_KEY")) # Set the API key in the environment variable


"""Embedding model"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5", 
    device="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True,
    cache_folder="cache",
)


"""RAG Settings"""

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

# Define the LLM, embedding model and other parameters
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SemanticSplitterNodeParser(
        buffer_size=5,
        include_prev_next_rel=True,
        embed_model=embed_model,
    )
Settings.num_output = 256
Settings.context_window = 4096 # Maximum size of the input query

# Prompt engineering
def prompt(query):
    return "<s> [INST] <<SYS>>You are a finance assistant and advisor, your role is to provide simple, clear and concise response to the user question. If the question seems complex, try to resolve it step by step. If you don't know how to respond, just say it, don't try to create or imagine false information. You can use markdown syntax to enhance the look and readability of your answers.<</SYS>>\n\n" + query + "[/INST] "

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


""" Vector Database """

# Cector database class to store the index
class VectorDatabase:
    # Intialize the vector database from the existing index
    def __init__(self):
        if not os.path.exists(persist_dir):
            # Load the documents and create the index
            documents = SimpleDirectoryReader(path).load_data()
            nodes = Settings.node_parser.get_nodes_from_documents(documents) # split documents into chunks depending
            index = VectorStoreIndex(nodes)
            # Store it for later
            index.storage_context.persist(persist_dir=persist_dir)
        # Load the index from the storage
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage_context)
        print("Index loaded successfully")

    # Reset the index
    def reset(self):
        # Delete the existing index
        if os.path.exists(persist_dir):
            os.remove(persist_dir)
        # Load the documents and create the index
        self.__init__()
    

""" RAG Chatbot """

# RAG chatbot class to respond to the user queries
class RAG:
    # Initialize the RAG chatbot
    def __init__(self, vector_database):
        # Configure retriever
        self.retriever = VectorIndexRetriever(
            index=vector_database.index,
            similarity_top_k=15,
            verbose=False,
        )
        # Configure response synthesizer
        self.response_synthesizer = get_response_synthesizer(
            response_mode="compact",
        )
        # Assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
        )
        # Memory of the chatbot
        self.chat_history = []
        print("RAG chatbot loaded successfully")
    
    # Respond to the user query
    def respond(self, message):
        response = markdown.markdown(str(self.query_engine.query(prompt(message))))
        self.chat_history.append({"user": "User", "text": message})
        self.chat_history.append({"user": "Assistant", "text": response})

    # Get the chat history
    def get_chat_history(self):
        return self.chat_history

    # Reset the chat history
    def reset_history(self):
        self.chat_history = []