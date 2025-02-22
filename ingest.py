import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from models import Models
import time
load_dotenv()
model = Models()

# models 
embedding = model.embedding
chat = model.chat

# embedding constants
embedding_data = './data'
chunk_size = 1000
chunk_overlap = 50
chunk_interval = 10

vector_store = Chroma(
    collection_name='document',
    embedding_function=embedding,
    persist_directory='./db/chroma_db'
)

def ingest_file(file_path:str):
    if not file_path.lower().endswith('.pdf'):
        raise ValueError('Only PDF files are supported')
    loader = PyPDFLoader(file_path)
    load_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n'," ","."]
    )
    documents = text_splitter.split_documents(load_document)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

def main():
    while True:
        for filename in os.listdir(embedding_data):
            if not filename.startswith('ingest_'):     
                file_path = os.path.join(embedding_data, filename)
                ingest_file(file_path)
                new_filename = "ingest_" + filename
                os.rename(file_path, os.path.join(embedding_data, new_filename))
        time.sleep(10)

if __name__=="__main__":
    main()