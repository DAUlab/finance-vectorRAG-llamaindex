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

from groq import Groq as Groq_API
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

client = Groq_API(api_key=os.environ.get('GROQ_KEY'))

# Prompt engineering
def prompt(query, history):
    return "You are a finance assistant and advisor, your role is to provide clear and concise but detailled response to the user question.\n If the question seems complex, try to resolve it step by step. If you don't know how to respond, just say it, don't try to create or imagine false information.\n Use markdown syntax to enhance the look and readability of your answers.\n\n" + "Conversation history:\n" + history + "\n\nUser question:\n" + query

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


""" Vector Database """

# Cector database class to store the index
class VectorDatabase:
    # Intialize the vector database from the existing index
    def __init__(self):
        # Llama-index settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SemanticSplitterNodeParser(
                buffer_size=5,
                include_prev_next_rel=True,
                embed_model=embed_model,
            )
        Settings.num_output = 256
        Settings.context_window = 4096 # Maximum size of the input query
        self.settings = Settings
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
    def __init__(self, vector_database, chat_history="", messages=[]):
        self.vector_database = vector_database
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=vector_database.index,
            similarity_top_k=15,
            verbose=False,
        )
        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
        )
        # Assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        # Memory of the chatbot
        self.chat_history = chat_history
        self.messages = messages
        print("RAG chatbot loaded successfully")
    
    
    # Switching model
    def switch_model(self, model):
        self.vector_database.__init__(llm=Groq(model=model, api_key=os.environ.get('GROQ_KEY')))
        self.__init__(self.vector_database, self.chat_history, self.messages)
        print("Switched to", self.vector_database.settings.llm.model)

    # Chat history summarization
    def history_input(self, query):
        history_summarizer = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "Just summarize the given conversation, it will be used as context for the ongoing conversation. Focus on the user informations and the last assistant message. You are the assistant.\n\n" + str(self.chat_history),
            }], model="llama3-8b-8192", temperature=0.2)
        contextualized_query = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "History: \n\n" + str(self.chat_history) + "\n\n\nUser message:\n\n" + query + "\n\n SYSTEM INSTRUCTIONS: If the user question is not higly related to the conversation history JUST REPEAT THE USER MESSAGE AS IS. If the query require context from the history slightly rephrase it into a concise standalone question. Do NOT answer the question.\n\n",
            }], model="llama3-8b-8192", temperature=0.2)
        self.chat_history = history_summarizer.choices[-1].message.content
        enhanced_query = contextualized_query.choices[-1].message.content
        print('chat history: ' + self.chat_history + '\n------------')
        print('query: ' + enhanced_query)
        return enhanced_query
    
    # Respond to the user query
    def respond(self, message):
        self.chat_history += "\n\nUser:" + message
        query = self.history_input(message)
        response = markdown.markdown(str(self.query_engine.query(prompt(query, self.chat_history))))
        self.chat_history += "\n\nAssistant:" + response
        self.messages.append({"user": "User", "text": message})
        self.messages.append({"user": "Assistant", "text": response})

    # Reset the chat history
    def reset_history(self):
        self.chat_history = ""
        self.messages = []