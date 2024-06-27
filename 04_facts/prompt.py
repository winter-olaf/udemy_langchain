from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = 1

# load .env
load_dotenv()
API_KEY = os.environ.get('API_KEY')
chat = ChatOpenAI(
    openai_api_key=API_KEY)

embeddings = OpenAIEmbeddings(
    openai_api_key=API_KEY,
)
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

# retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("what is an interesting fact about the english language?")
print(result)
