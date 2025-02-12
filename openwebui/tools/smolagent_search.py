"""
title: SmolAgent Tool
author: Wanis Elabbar
author_url: https://github.com/elabbarw
date: 2024-02-09
version: 0.1.3
license: MIT
description: This tool uses SmolAgents to generate answers by writing and running code in E2B by default.
requirements: smolagents[e2b]
"""

from typing import Any, Awaitable, Callable, Optional
from functools import partial
from pydantic import BaseModel, Field
import asyncio
import os

# Import smolagents dependencies.
from smolagents import (
    CodeAgent,
    OpenAIServerModel,
    VisitWebpageTool,
    DuckDuckGoSearchTool
)


class Tools:
    class Valves(BaseModel):
        # Add any valves parameters if needed.
        OPENAI_BASE_URL: str = Field(default="", description="OpenAI Base URL")
        OPENAI_API_KEY: str = Field(default="", description="OpenAI Key")
        OPENAI_MODEL: str = Field(default="", description="Model to use")
        E2B_KEY: Optional[str] = Field(default="", description="E2B API Key")
        E2B_DOMAIN: Optional[str] = Field(default="", description="E2B Domain if you're self-hosting")
        E2B_MODE: Optional[bool] = Field(default=True, description="Run the Agent in an E2B Environment (Safer!)")
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def smolagent_search(
        self,
        message_context: str,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Contact an AI Agent to Perform Research on the Query
        :param: The request and any relevant context
        :return: Results of the research
        """

        try:
            if self.valves.E2B_MODE and self.valves.E2B_KEY:
                os.environ['E2B_API_KEY'] = self.valves.E2B_KEY
                if self.valves.E2B_DOMAIN:
                    os.environ['E2B_DOMAIN'] = self.valves.E2B_DOMAIN

            # Create the OpenAI model using configuration values.
            model = OpenAIServerModel(
                api_base=self.valves.OPENAI_BASE_URL,
                api_key=self.valves.OPENAI_API_KEY,
                model_id=self.valves.OPENAI_MODEL,
            )

            def initialize_and_run_agent(message_context):
                
                agent = CodeAgent(
                    tools=[DuckDuckGoSearchTool(),VisitWebpageTool()],
                    model=model,
                    additional_authorized_imports=[
                    ],
                    max_print_outputs_length=200,
                    use_e2b_executor=True if self.valves.E2B_MODE and self.valves.E2B_KEY else False
                    
                )
                return agent.run(message_context, reset=False)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Researching...",
                        "done": False,
                    },
                }
            )
            # Run the agent initialization and execution in a separate thread
            result = await asyncio.to_thread(partial(initialize_and_run_agent, message_context))


            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Research Complete!",
                        "done": True,
                    },
                }
            )
            return result

        except Exception as e:
            error_message = f"SmolAgents Error: {str(e)}"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": error_message,
                        "done": True,
                    },
                }
            )
            return error_message


async def main():
    tools = Tools()
    
    # Set up the required environment variables
    tools.valves.OPENAI_BASE_URL = ""
    tools.valves.OPENAI_API_KEY = ""
    tools.valves.OPENAI_MODEL = ""
    tools.valves.E2B_KEY = ""
    tools.valves.E2B_DOMAIN = ""  # Optional
    tools.valves.E2B_MODE = True

    # Define a simple event emitter function
    async def event_emitter(event):
        print(f"Event: {event}")

    # Test the smolagent_search method
    message_context = "Where does Marcus Aurelius come from?"
    result = await tools.smolagent_search(message_context, event_emitter)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())