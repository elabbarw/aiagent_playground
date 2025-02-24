"""
title: Chat With SmolAgents Pipeline
author: elabbarw
author_url: https://github.com/elabbarw
git_url: https://github.com/elabbarw/aiagent_playground/blob/main/openwebui/pipelines/chat_with_smolagents.py
date: 2024-02-11
version: 0.1.0
license: MIT
description: A basic Pipeline for chatting with SmolAgents
requirements: smolagents[e2b]
"""

import os
from typing import Optional, Union, Generator, Iterator
from pydantic import BaseModel, Field

from smolagents import (
    CodeAgent,
    OpenAIServerModel,
    VisitWebpageTool,
    DuckDuckGoSearchTool
)

class Pipeline:
    class Valves(BaseModel):
        OPENAI_BASE_URL: str = Field(default="", description="OpenAI Base URL")
        OPENAI_API_KEY: str = Field(default="", description="OpenAI Key")
        OPENAI_MODEL: str = Field(default="", description="Model to use")
        E2B_KEY: Optional[str] = Field(default="", description="E2B API Key")
        E2B_DOMAIN: Optional[str] = Field(default="", description="E2B Domain if you're self-hosting")
        E2B_MODE: Optional[bool] = Field(default=False, description="Run the Agent in an E2B Environment (Safer!)")
        pass

    def __init__(self):
        self.name = "Chat With SmolAgents"
        self.valves = self.Valves()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the  request is made. You can modify the form data before it is sent.

        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the  response is completed. You can modify the messages after they are received.


        return body
    
    def pipe(
        self,
        body: dict,
        messages: list[dict],
        user_message: str,
        model_id: str,
    ) -> Union[str, Generator, Iterator]:
        
        if self.valves.E2B_MODE and self.valves.E2B_KEY:
            os.environ['E2B_API_KEY'] = self.valves.E2B_KEY
            if self.valves.E2B_DOMAIN:
                os.environ['E2B_DOMAIN'] = self.valves.E2B_DOMAIN
        
        incomingmessages = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
        
        model = OpenAIServerModel(
                    api_base=self.valves.OPENAI_BASE_URL,
                    api_key=self.valves.OPENAI_API_KEY,
                    model_id=self.valves.OPENAI_MODEL,
                )

        try:
            agent = CodeAgent(
                            tools=[DuckDuckGoSearchTool(),VisitWebpageTool()],
                            model=model,
                            additional_authorized_imports=[],
                            use_e2b_executor=True if self.valves.E2B_KEY and self.valves.E2B_MODE else False
                        )

            result = agent.run(incomingmessages, reset=False)

            yield result
        except Exception as e:
            yield str(e)
