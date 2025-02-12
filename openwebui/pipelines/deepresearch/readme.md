# Open WebUI - SmolAgents Open Deep Research Pipeline

To manage the resource-intensive requirements, it's optimal to host this within the Pipelines container. Follow these steps to launch:

1. Save your API key in a `.env` file.
2. Execute `docker compose up -d --build` to start the service.
3. Navigate to Open WebUI Admin -> Pipelines.
4. Add a new connection pointing to `http://host.docker.internal:9099`, using the API key saved earlier.
5. Upload the `smolagent_deepresearch.py` file. 
6. Enter the required LLM information during this step to ensure proper deployment. This will set it up as a model in the system.

## Special Thanks

This project leverages the work from `smolagents`, a lightweight library designed to build sophisticated agentic systems and the `open deep research` project. Special thanks to:

- Aymeric Roucher
- Albert Villanova del Moral
- Thomas Wolf
- Leandro von Werra
- Erik Kaunismäki

For more information, visit the [smolagents GitHub repository](https://github.com/huggingface/smolagents).
