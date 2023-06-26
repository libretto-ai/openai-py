#!/usr/bin/env python
import os
import sys
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import im_openai

im_openai.patch_openai()

from langchain.llms import OpenAI, OpenAIChat

llm = OpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    verbose=True,
    model_kwargs={"ip_project_key": "alecf-local-playground"},
)

socks = llm.predict(
    "What would be a good company name for a company that makes colorful socks?",
)

print("socks = ", socks)

llm2 = OpenAIChat(openai_api_key=os.environ["OPENAI_API_KEY"], verbose=True)

socks2 = llm2.predict(
    "What would be a good company name for a company that makes colorful socks?",
    ip_event_id=str(uuid.uuid4()),
)

print("socks2 = ", socks2)
