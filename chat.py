import os

from models import Models
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
model = Models()
embedding = model.embedding
chat = model.chat
# prompt template

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {input}

Start the answer directly. No small talk please.
"""

def custom_prompt(custom_prompt_template:str):
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the data provided."),
        ("human", custom_prompt_template)
    ])

# retrieval qa
DB_PATH = "./db/chroma_db"
vector_store = Chroma(
    collection_name='documents',
    embedding_function=embedding,
    persist_directory=DB_PATH
)

retriever = vector_store.as_retriever(kwargs={"k":10})
combine_docs_chain = create_stuff_documents_chain(
    chat,
    custom_prompt(CUSTOM_PROMPT_TEMPLATE),
)

retrieval_chain = create_retrieval_chain(retriever,combine_docs_chain)

def main():
    while True:
        query = input("User (or type 'q','quit','exit' to exit): ")
        if query.lower() in ['q','quit','exit']:
            break
        result = retrieval_chain.invoke({"input": query})
        print("Assistant: " ,result['answer'],"\n\n")

if __name__ == '__main__':
    main()