from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# load .env
load_dotenv()
API_KEY = os.environ.get('API_KEY')

llm = OpenAI(
    openai_api_key=API_KEY,
)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)


code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

# result = code_chain({
#     "language": args.language,
#     "task": args.task
# })

result = chain({
    "language": args.language,
    "task": args.task
})

# print(result["text"])
print(">>>>>>>> GENERATED CODE <<<<<<<<")
print(result["code"])

print(">>>>>>>> GENERATED TEST <<<<<<<<")
print(result["test"])

