import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# --- Imports for LCEL ---
# We replace the create_..._chain imports with these building blocks
from operator import itemgetter  # A handy tool to get a value from a dict
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
# --- End of new imports ---

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate

# --- 1. SETUP (Your code is perfect) ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# Initialize your models and retriever
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", google_api_key=api_key
)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# Your prompt template
template = """
You are a helpful AI assistant. Answer the user's question based on the
following context. If you don't know the answer, just say "I don't know."

Context: {context}
Question: {input}
"""
prompt = PromptTemplate.from_template(template)


# --- 2. BUILD YOUR CHAIN WITH LCEL ---

# This is the equivalent of 'create_stuff_documents_chain'
# It "stuffs" the context and input into the prompt, then calls the model.
document_chain = prompt | llm | StrOutputParser()

# This is the equivalent of 'create_retrieval_chain'
# It defines the full RAG process.
retrieval_chain = RunnableParallel(
    # "context": Run the retriever on the user's "input"
    context=(itemgetter("input") | retriever), 
    # "input": Pass the user's "input" straight through
    input=itemgetter("input")
) | document_chain # Pipe the resulting {context, input} dict into our document_chain


# --- 3. YOUR API (No changes needed) ---

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask_query(query: Query):
    # Use the .invoke() method on your new LCEL chain
    # It expects a dictionary matching the 'itemgetter' keys
    response = retrieval_chain.invoke({"input": query.query})
    
    return {"answer": response}