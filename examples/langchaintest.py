#!/usr/bin/env python
import os
import sys
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from libretto_openai import patch_openai, LibrettoCreateParams

patch_openai(allow_unnamed_prompts=True)

from langchain.llms import OpenAI, OpenAIChat

llm = OpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    verbose=True,
    model_kwargs={
        "libretto": LibrettoCreateParams(
            project_key="alecf-local-playground",
        ),
    },
)

try:
    socks = llm.predict(
        "What would be a good company name for a company that makes really colorful socks?",
    )
    print("socks = ", socks)
except Exception as e:
    print("Error:", e)

llm2 = OpenAIChat(openai_api_key=os.environ["OPENAI_API_KEY"], verbose=True)

socks2 = llm2.predict(
    "What would be a good company name for a company that makes very colorful socks?",
    libretto=LibrettoCreateParams(
        event_id=str(uuid.uuid4()),
    ),
)

print("socks2 = ", socks2)
