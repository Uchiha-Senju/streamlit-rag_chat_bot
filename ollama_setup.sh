#!/bin/bash

# Start ollama
ollama serve &
# Wait for it
sleep 10

# Download the models into the image
ollama pull gemma2:2b
ollama pull nomic-embed-text
