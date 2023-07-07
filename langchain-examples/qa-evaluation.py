"""Demo for simple QA evaluation.

To use, you need to pip install a few modules:
    pip install faiss-cpu tiktoken
"""
import logging
import os
import sys

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from im_openai import langchain_util

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("im_openai")
logger.setLevel(logging.INFO)

callbacks = langchain_util.PromptWatchCallbacks(
    project_key="example-langchain-simple-qa", api_name="qa-evaluation"
)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
loader = TextLoader("./state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(client=ChatOpenAI(client=ChatCompletion))
docsearch = FAISS.from_documents(texts, embeddings)
qa = RetrievalQA.from_llm(
    llm=ChatOpenAI(client=ChatCompletion, callbacks=[callbacks]),
    retriever=docsearch.as_retriever(),
    callbacks=[callbacks],
)

print(qa({"query": "What is the state of the union?"}))
