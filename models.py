from langchain_ollama import OllamaEmbeddings,ChatOllama

class Models:
    def __init__(self) -> None:
        self.embedding = OllamaEmbeddings(
            model='mxbai-embed-large'
        )
        self.chat = ChatOllama(
            model='lama3.2:3b',
            temperature=0
        )