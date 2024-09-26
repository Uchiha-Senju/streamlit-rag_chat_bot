FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
    
RUN git clone https://github.com/Uchiha-Senju/streamlit-rag_chat_bot.git .

RUN pip3 install -r requirements.txt
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN chmod +x ollama_setup.sh
RUN chmod +x run_app.sh
RUN ./ollama_setup.sh

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["./run_app.sh"]