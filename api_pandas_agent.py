import ast
import json
from typing import Any, Optional
import pandas as pd
from langchain import OpenAI
from langchain.agents import Tool
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.requests import RequestsWrapper
from dotenv import load_dotenv
import os
from pandas import DataFrame
from pydantic import BaseModel
from prompt import PREFIX, SUFFIX

load_dotenv()


class CampAgent(BaseModel):
    df: Optional[DataFrame] = None
    requests_wrapper: Optional[RequestsWrapper] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.requests_wrapper = RequestsWrapper(headers={
            "x-api-key": os.getenv("DOC_API_KEY")
        })
        self.df = pd.DataFrame()

    def find(self, name) -> str:
        res = self.requests_wrapper.get('https://api.doc.govt.nz/v2/campsites')

        df = pd.DataFrame(json.loads(res))
        self.df = df

        if len(df) == 0:
            return "I can't find any campsites"

        if len(df) > 50:
            return str(df.columns)

        return df.to_string()

    def filter(self, query) -> str:
        globals_df = {"df": self.df}
        try:
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), globals_df, globals_df)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            try:
                return eval(module_end_str, globals_df)
            except Exception:
                exec(module_end_str, globals_df)
                return ""
        except Exception as e:
            return str(e)

    def detail(self, asset_id: str) -> str:
        print(asset_id)
        return self.requests_wrapper.get(
            f'https://api.doc.govt.nz/v2/campsites/{asset_id}/detail')


def campsite_pandas_agent(
        llm: BaseLLM,
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        verbose: bool = False,
        **kwargs: Any,
) -> AgentExecutor:
    input_variables = ["input", "agent_scratchpad", "chat_history"]
    hut = CampAgent()

    tools = [
        Tool(
            name="find",
            func=hut.find,
            description=("A tool for checking the campsites."
                         "It returns a list of unfiltered campsites."
                         "Use contains to check if a string is in the output."
                         "If you see 'I can't find any campsites', say that"
                         )
        ),
        Tool(
            name="python_repl_ast",
            func=hut.filter,
            description=(
                "A Python shell. Use this to execute python commands. "
                "Input should be a valid python command. "
                "When using this tool, sometimes output is abbreviated - "
                "make sure it does not look abbreviated before using it in your answer."
                "Use df to check if a dataframe is in the output."
                "If not say I can't find it"
            )
        ),
        Tool(
            name="detail",
            func=hut.detail,
            description=("A tool for retrieving details."
                         "Must be used after filter."
                         "It takes an input of asset_id, and returns the details of the"
                         "campsite."
                         "You will need to use python repl first to get the asset_id."
                         "If you see 'Invalid campsite id', say that"
                         )
        ),
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
    agent = campsite_pandas_agent(
        OpenAI(temperature=0, openai_api_key=os.getenv("OPEN_API_KEY")),
        verbose=True)
    print(agent.run(input="Routeburn"))
