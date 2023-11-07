import logging
import os
import sys

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import ChatCompletion

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from libretto_openai import langchain as langchain_util

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("libretto_openai")
logger.setLevel(logging.INFO)

with langchain_util.prompt_watch_tracing(
    api_key=os.getenv("LIBRETTO_API_KEY"),
    prompt_template_name=os.path.basename(__file__),
):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )

    document_body = ""
    with open(os.path.join(os.path.dirname(__file__), "state_of_the_union.txt")) as f:
        document_body = f.read()

    texts = text_splitter.create_documents([document_body])

    llm = ChatOpenAI(client=ChatCompletion)
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    print("Found ", len(texts), " paragraphs.")
    summary = chain(texts[0:4])
    print(summary["output_text"])

    import time

    time.sleep(5)
