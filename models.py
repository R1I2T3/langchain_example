from langchain_ollama import OllamaEmbeddings,ChatOllama

class Models:
    def __init__(self) -> None:
        self.embedding = OllamaEmbeddings(
            model='mxbai-embed-large'
        )
        self.chat = ChatOllama(
            model='gemma:2b',
            temperature=0
        )