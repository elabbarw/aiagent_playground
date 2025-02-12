# Use the existing Pipelines image as the base image
FROM ghcr.io/open-webui/pipelines:main

# Set the working directory
WORKDIR /app

# Copy the scripts into the folder
COPY scripts ./scripts

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt