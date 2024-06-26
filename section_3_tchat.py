from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory

# load .env
load_dotenv()
API_KEY = os.environ.get('API_KEY')

chat = ChatOpenAI(
    openai_api_key=API_KEY,
    # verbose=True
)

# return_messages: message의 string만 던지는 것이 아니라 객체를 넘겨서 메시지의 정보를 더 활용할 수 있도록 함
# memory = ConversationBufferMemory(
#     memory_key="messages",
#     return_messages=True,
#     chat_memory=FileChatMessageHistory("chat_history.json")
# )
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        # variable_name: MsgPlaceholder to go and look at out input variable
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while 1:
    content = input("Enter: >> ")
    result = chain({"content": content})
    print(result["text"])
    # print(memory.buffer)
