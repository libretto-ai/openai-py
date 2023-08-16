"""Demo for simple QA evaluation.

To use, you need to pip install a few modules:
    pip install faiss-cpu tiktoken
"""
import logging
import os
import sys
import uuid

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from im_openai import langchain as langchain_util

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("im_openai")
logger.setLevel(logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

with langchain_util.prompt_watch_tracing(
    "f1ed34de-5069-48f9-a513-6095c45e3a30",
    api_name=os.path.basename(__file__),
    chat_id=str(uuid.uuid4()),
):
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "state_of_the_union.txt"))

    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(client=ChatOpenAI(client=ChatCompletion))
    docsearch = FAISS.from_documents(texts, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(client=ChatCompletion),
        retriever=docsearch.as_retriever(),
        memory=memory,
    )

    print(qa({"question": "What is the state of the union?"}))
    print(qa({"question": "And who said that?"}))
    print(qa({"question": "Is that person still alive?"}))
