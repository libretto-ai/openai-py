import logging
import os
import sys

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from openai import ChatCompletion

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from im_openai import langchain as langchain_util

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("im_openai")
logger.setLevel(logging.INFO)

callbacks = langchain_util.PromptWatchCallbacks(
    api_key="f1ed34de-5069-48f9-a513-6095c45e3a30", api_name="math-qa-2"
)

llm = ChatOpenAI(client=ChatCompletion, temperature=0, callbacks=[callbacks])
tools = load_tools(["pal-math"], llm=llm, callbacks=[callbacks])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True,
    # callbacks=[callbacks],
    # agent_kwargs={"callbacks": [callbacks]},
)

result = agent.run(
    "If my age is half of my dad's age and he is going to be 60 next year, what is my current age?",
    callbacks=[callbacks],
)

print("Agent Result: ", result)
