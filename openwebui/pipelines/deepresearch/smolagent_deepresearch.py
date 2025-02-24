"""
title: SmolAgents Deep Research Pipeline
author: elabbarw, aymeric-roucher & albertvillanova
author_url: https://github.com/elabbarw
original_author_url: https://github.com/huggingface/smolagents
git_url: https://github.com/elabbarw/aiagent_playground/blob/main/openwebui/pipelines/deepresearch/smolagent_deepresearch.py
date: 2024-02-11
version: 0.1.0
license: MIT
description: A pipeline to kick off deep research - install requirements directly in pipelines shell or through docker-compose
"""
import os
import threading
from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator

from dotenv import load_dotenv
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    OpenAIServerModel,
)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)


append_answer_lock = threading.Lock()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"


class Pipeline:
    class Valves(BaseModel):
        # Add any valves parameters if needed.
        OPENAI_BASE_URL: str = Field(
            default="", description="OpenAI Base URL")
        OPENAI_API_KEY: str = Field(
            default="", description="OpenAI Key")
        OPENAI_MODEL: str = Field(
            default="", description="Model to use")
        SERPAPI_API_KEY: Optional[str] = Field(
            default="", description="The SERP API Key if required"
        )
        BING_API_KEY: Optional[str] = Field(
            default="", description="The BING API Key if required"
        )
        GOOGLE_API_KEY: Optional[str] = Field(
            default="", description="The Google Search API Key if required"
        )
        GOOGLE_API_CX: Optional[str] = Field(
            default="", description="The Google Search CX Key if required"
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.BROWSER_CONFIG = {
                    "viewport_size": 1024 * 5,
                    "downloads_folder": "downloads_folder",
                    "request_kwargs": {
                        "headers": {"User-Agent": user_agent},
                        "timeout": 300,
                    },
                    "serpapi_key": self.valves.SERPAPI_API_KEY,
                    "bingapi_key": self.valves.BING_API_KEY,
                    "googleapi_key": self.valves.GOOGLE_API_KEY,
                    "googleapi_cx": self.valves.GOOGLE_API_CX
                }

        os.makedirs(f"./{self.BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


    def pipe(self,
        body: dict,
        messages: list[dict],
        user_message: str,
        model_id: str,
    ) -> Union[str, Generator, Iterator]:
        
        incomingmessages = "\n".join([f"{message['role']}: {message['content']}" for message in messages])

        text_limit = 100000

        model = OpenAIServerModel(
                api_base=self.valves.OPENAI_BASE_URL,
                api_key=self.valves.OPENAI_API_KEY,
                model_id=self.valves.OPENAI_MODEL
            )
        document_inspection_tool = TextInspectorTool(model, text_limit)

        browser = SimpleTextBrowser(**self.BROWSER_CONFIG)

        WEB_TOOLS = [
            SearchInformationTool(browser),
            VisitTool(browser),
            PageUpTool(browser),
            PageDownTool(browser),
            FinderTool(browser),
            FindNextTool(browser),
            ArchiveSearchTool(browser),
            TextInspectorTool(model, text_limit),
        ]
        text_webbrowser_agent = ToolCallingAgent(
            model=model,
            tools=WEB_TOOLS,
            max_steps=20,
            verbosity_level=2,
            planning_interval=4,
            name="search_agent",
            description="""A team member that will search the internet to answer your question.
        Ask him for all your questions that require browsing the web.
        Provide him as much context as possible, in particular if you need to search on a specific timeframe!
        And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
        Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
        """,
            provide_run_summary=True,
        )
        text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

        manager_agent = CodeAgent(
            model=model,
            tools=[visualizer, document_inspection_tool],
            max_steps=12,
            verbosity_level=2,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            planning_interval=4,
            managed_agents=[text_webbrowser_agent],
        )

        answer = manager_agent.run(incomingmessages)

        yield f"Got this answer: {answer}"
