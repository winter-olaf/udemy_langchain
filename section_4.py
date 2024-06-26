from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os

# load .env
load_dotenv()
API_KEY = os.environ.get('API_KEY')

embeddings = OpenAIEmbeddings(
    openai_api_key=API_KEY,
)

emb = embeddings.embed_query("hi there")

print(emb)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=100 # 텍스트 앞뒤로 겹침
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter,
)

# for doc in docs:
#     print(f"{doc.page_content}\n")

