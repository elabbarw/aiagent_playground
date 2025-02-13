"""
title: LiteLLM Stable Diffusion Image Generation Action
author: Wanis Elabbar
author_url: https://github.com/elabbarw
date: 2025-02-12
version: 0.1.2
license: MIT
description: This action generates an image using SD models deployed on AWS Bedrock and presented via LiteLLM.
"""

# Personally i have a normal GPT4O model with system prompts to generate appropriate SD image prompts. Once the user is happy with the prompt they click on the action button.

import asyncio
import base64
import uuid
import re
import json
import mimetypes
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


from open_webui.config import CACHE_DIR
import requests


class Action:
    class Valves(BaseModel):
        LITELLM_API_KEY: str = Field(
            default="your_api_key_here", description="Required API key for LiteLLM"
        )
        LITELLM_IMAGE_URL: str = Field(
            default="https://[your litellm gateway].com/images/generations",
            description="LiteLLM Endpoint image generation",
        )
        pass

    def __init__(self):
        # You can set these either here or via environment variables.
        self.valves = self.Valves()
        self.IMAGE_CACHE_DIR = Path(CACHE_DIR).joinpath("./image/generations/")
        self.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        ### Put LiteLLM names here
        self.modelnames = {
            "sdxl": "eit_sdxl",
            "core": "eit_sdcore",
            "large3": "eit_sd3large",
            "ultra": "eit_sdultra",
            "large35": "eit_sd35large",
        }
        pass

    def save_b64_image(self, b64_str: str) -> str:
        try:
            image_id = str(uuid.uuid4())

            if "," in b64_str:
                header, encoded = b64_str.split(",", 1)
                mime_type = header.split(";")[0]

                img_data = base64.b64decode(encoded)
                image_format = mimetypes.guess_extension(mime_type)

                image_filename = f"{image_id}{image_format}"
                file_path = self.IMAGE_CACHE_DIR / f"{image_filename}"
                with open(file_path, "wb") as f:
                    f.write(img_data)
                return image_filename
            else:
                image_filename = f"{image_id}.png"
                file_path = self.IMAGE_CACHE_DIR.joinpath(image_filename)

                img_data = base64.b64decode(b64_str)

                # Write the image data to a file
                with open(file_path, "wb") as f:
                    f.write(img_data)
                return image_filename

        except Exception as e:
            raise Exception(f"Error saving image: {e}")

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        try:

            response = await __event_call__(
                {
                    "type": "input",
                    "data": {
                        "title": "Enter the SD Model (sdxl, core, large3, ultra, large35)",
                        "message": "$0.04, $0.04, $0.08, $0.08, $0.14",
                        "placeholder": "Enter the model name",
                    },
                }
            )

            if not response or response not in self.modelnames:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "You didn't pick a model!",
                            "done": True,
                        },
                    }
                )
                return

            modelchoice = self.modelnames[response]

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Generating Stable Diffusion Image...",
                            "done": False,
                        },
                    }
                )

            last_message = body["messages"][-1]
            prompt = last_message["content"]
            

            # Regular expression to capture text after 'NEGATIVE:' (if any)
            negmatch = re.search(r"(?i)negative:?\s*(.*)", prompt)
            negmatch_string = None
            if negmatch:
                negmatch_string = negmatch.group(1).strip()


            headers = {
                "X-API-KEY": self.valves.LITELLM_API_KEY,
                "Content-Type": "application/json",
            }

            if "sdxl" in modelchoice:
                payload =  {
                    "prompt": str(prompt),
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "samples": 1,
                    "steps": 30,
                    "response_format": "b64_json",
                    "model": str(modelchoice)
                }
            else:
                payload = {
                    "prompt": str(prompt),
                    "negative_prompt": str(negmatch_string),
                    "mode": "text-to-image",
                    "model": str(modelchoice),
                    "aspect_ratio": "1:1",
                    "response_format": "b64_json"
                }

            payload.update(
                {"metadata": {
                        "tags": [
                            "openwebui",
                            str(modelchoice),
                            (
                                __user__["email"]
                                if __user__ and "email" in __user__
                                else "unknown"
                            ),
                            (
                                __user__["name"]
                                if __user__ and "name" in __user__
                                else "unknown"
                            ),
                        ]
                    }}
                    )

            response = await asyncio.to_thread(
                requests.post,
                self.valves.LITELLM_IMAGE_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            response_data = response.json()
            # Check if the response structure is as expected
            if not isinstance(response_data, dict) or "data" not in response_data:
                raise Exception(f"Unexpected response format: {response_data}")

            images = []
            for image in response_data["data"]:
                image_filename = self.save_b64_image(image["b64_json"])
                images.append({"url": f"/cache/image/generations/{image_filename}"})
                file_body_path = self.IMAGE_CACHE_DIR.joinpath(f"{image_filename}.json")

                with open(file_body_path, "w") as f:
                    json.dump(payload, f)

            # Emit each image as a message
            for image in images:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": f"![Generated Image]({image['url']})\n"},
                    }
                )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Image generated successfully",
                            "done": True,
                        },
                    }
                )

        except Exception as e:
            error_message = f"Error generating image: {str(e)}"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": error_message,
                        "done": True,
                    },
                }
            )

        return
