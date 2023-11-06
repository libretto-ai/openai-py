"""Demo for simple QA evaluation.

To use, you need to pip install a few modules:
    pip install faiss-cpu tiktoken
"""
import logging
import os
import pprint
import shutil
import sys
import uuid

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from libretto_openai import langchain as langchain_util

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("libretto_openai")
logger.setLevel(logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

ip_prompt_template_name = os.path.basename(__file__)
with langchain_util.prompt_watch_tracing(
    api_key="f1ed34de-5069-48f9-a513-6095c45e3a30",
    prompt_template_name=ip_prompt_template_name,
    chat_id=str(uuid.uuid4()),
):
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "state_of_the_union.txt"))

    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(client=ChatOpenAI())
    docsearch = FAISS.from_documents(texts, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    condense_question_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Given the current conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Follow Up Input: {question}
Standalone question:"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )
    # super hack here: langchain fix in https://github.com/langchain-ai/langchain/pull/9555
    # Hoping for fix in or around 0.272.0+, then we can just pass this directly to `from_messages()`
    if "additional_kwargs" not in condense_question_template._lc_kwargs:
        condense_question_template._lc_kwargs["additional_kwargs"] = {}
    condense_question_template._lc_kwargs["additional_kwargs"].update(
        ip_prompt_template_name=f"{ip_prompt_template_name}/condense_question"
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=docsearch.as_retriever(),
        get_chat_history=lambda m: m,
        condense_question_prompt=condense_question_template,
        memory=memory,
    )

    width = shutil.get_terminal_size().columns
    pprint.pp(qa({"question": "What is the state of the union?"}), width=width)
    pprint.pp(qa({"question": "And who said that?"}), width=width)
    pprint.pp(qa({"question": "Is that person still alive?"}), width=width)
