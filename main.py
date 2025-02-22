from fastapi import FastAPI
from pydantic import BaseModel
from chat import retrieval_chain
class Input(BaseModel):
    human_input:str

class Output(BaseModel):
    output:str

app = FastAPI()

@app.post("/chat")
def chat(input:Input):
    return retrieval_chain.invoke({"input":input.human_input})

