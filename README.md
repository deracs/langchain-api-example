# API agents for Campsite project
Stripped back version for twitter

Built with [LangChain](https://github.com/hwchase17/langchain/)
and [DOC - Api](https://api.doc.govt.nz/apis).

## 🚀 agents
### campsite_agent
`base.campsite_agent`
This agent does the filtering at request level using pandas. 
It is a simple implementation of the filtering logic. 

### campsite_pandas_agent (flakey)
`api_pandas_agent`
This agent use the python repl tool to do the filtering. Uses index columns from DF to do further
filtering. 

## ✅ Running locally
1. Install dependencies: `pip install -r requirements.txt`
2. Add environment variables to `.env` file