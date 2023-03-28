import json
from langchain.tools import BaseTool
from langchain.tools.requests.tool import BaseRequestsTool
import pandas as pd


class GetCampsitesTool(BaseRequestsTool, BaseTool):
    """Tool for making a GET request to an API endpoint."""

    name = "campsite_find"
    description = """A tool for checking the campsites.  
                    It takes a string as input and returns a list of possible 
                    campsites."""
    url = "https://api.doc.govt.nz/v2/campsites"

    def _run(self, name: str) -> str:
        """Run the tool."""
        res = self.requests_wrapper.get(self.url)
        df = pd.DataFrame(json.loads(res))

        df = df[df['name'].str.contains(name, case=False)]
        return df.to_string()

    async def _arun(self, name: str) -> str:
        """Run the tool asynchronously."""
        res = await self.requests_wrapper.aget(self.url)
        df = pd.DataFrame(json.loads(res))
        df = df[df['name'].str.contains(name, case=False)]
        return df.to_string()


class GetCampsiteTool(BaseRequestsTool, BaseTool):
    """Tool for making a GET request to an API endpoint."""

    name = "campsite_detail"
    description = """A tool for retrieving campsite details. It takes an input of 
    asset_id, and returns the details of the campsite. You will need to use 
    campsite_find first to get the asset_id."""
    url = "https://api.doc.govt.nz/v2/campsites/{asset_id}/detail"

    def _run(self, asset_id: str) -> str:
        """Run the tool."""
        return self.requests_wrapper.get(self.url.format(asset_id=asset_id))

    async def _arun(self, asset_id: str) -> str:
        """Run the tool asynchronously."""
        return await self.requests_wrapper.aget(self.url.format(asset_id=asset_id))
