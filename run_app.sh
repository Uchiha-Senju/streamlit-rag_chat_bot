#!/bin/bash

# Spin up ollama
ollama serve >/dev/null 2>&1 &
# Run the app
streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0