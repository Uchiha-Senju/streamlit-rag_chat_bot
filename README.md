# Local RAG Bot using Streamlit and Ollama

- Build the docker image using 
```bash
curl -o rag_bot.Dockerfile https://raw.githubusercontent.com/Uchiha-Senju/streamlit-rag_chat_bot/refs/heads/main/rag_bot.Dockerfile
docker build . -f rag_bot.Dockerfile -t rag_bot/v1
```
- Run the container using
```bash
docker run -p 8501:8501 rag_bot/v1
```
- Open the app in a browser by going to `https://localhost:8501`

This runs the LLMs locally, so either have GPUs or be prepared to wait.