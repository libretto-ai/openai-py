import logging
import os
import sys

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
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
    llm = ChatOpenAI(client=ChatCompletion, temperature=0)
    tools = load_tools(["llm-math"], llm=llm)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # verbose=True,
        # callbacks=[callbacks],
        # agent_kwargs={"callbacks": [callbacks]},
    )

    result = agent.run(
        "If my age is half of my dad's age and he is going to be 60 next year, what is my current age?"
    )

print("Agent Result: ", result)
