# flake8: noqa
PREFIX = """
You are working with an campsite api. You should use the tools below to answer the 
question posed of you:"""


SUFFIX = """When using the python repl for pandas. Use contains for searchin strings. 
If you see I can't find any campsites or Invalid campsite, just say can't find any 
campsites. Begin! {chat_history} Question: {input} {agent_scratchpad}"""
