"""Agent for working with pandas objects."""
from typing import Any, Optional

from langchain import OpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.requests import RequestsWrapper
from dotenv import load_dotenv
import os
from prompt import PREFIX, SUFFIX
from tools import GetCampsitesTool, GetCampsiteTool

load_dotenv()


def campsite_agent(
        llm: BaseLLM,
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        verbose: bool = False,
        **kwargs: Any,
) -> AgentExecutor:
    input_variables = ["input", "agent_scratchpad", "chat_history"]
    api = RequestsWrapper(headers={
        "x-api-key": os.getenv("DOC_API_KEY")
    })

    tools = [
        GetCampsitesTool(requests_wrapper=api),
        GetCampsiteTool(requests_wrapper=api),
    ]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
        memory=readonlymemory,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools,
                                              verbose=verbose,
                                              memory=memory)


if __name__ == "__main__":
    agent = campsite_agent(
        OpenAI(temperature=0, openai_api_key=os.getenv("OPEN_API_KEY")),
        verbose=True)
    print(agent.run(input="Routeburn"))
