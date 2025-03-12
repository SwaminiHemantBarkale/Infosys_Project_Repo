from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
from fastapi.middleware.cors import CORSMiddleware


warnings.filterwarnings("ignore", category= DeprecationWarning)

load_dotenv()
client= Groq(api_key= os.getenv("API_KEY"))

app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials= True,
    allow_methods= ["*"],
    allow_headers= ["*"],
)

class QuestionRequest(BaseModel):
    question:str

loader= PyPDFLoader('./health.pdf')
data= loader.load()

text_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text= text_splitter.split_documents(data)

embeddings= FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db= FAISS.from_documents(text, embeddings)
retriever= db.as_retriever(search_type='similarity', search_kwargs={'k':4})

prompt_template= """
    You are a helpful assistant who answers only based on the provided context.
    If you don't know, say so. Don't make up answers.
    Answer in a single line.

    Context:{context},
    Question:{question}
"""

prompt= PromptTemplate(template= prompt_template, input_variables=['context','question'])

llm = ChatGroq(model_name="llama3-70b-8192")
qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type='stuff',
                                retriever=retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": prompt})

@app.post("/")
def ask_question(request: QuestionRequest):
    result = qa(request.question)
    return {"answer": result['result']}




