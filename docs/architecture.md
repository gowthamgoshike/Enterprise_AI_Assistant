# System Architecture

## Data Platform

Transaction Generator
        |
        v
      Kafka
        |
        v
Spark Structured Streaming
        |
        v
AWS S3
Bronze → Silver → Gold
        |
        v
PostgreSQL

---

## AI Application

Authenticated User
        |
        v
Streamlit
        |
        v
FastAPI
        |
        v
Authentication / Authorization
        |
        v
AI Agent
        |
        +---- Tool Calling
        |
        +---- RAG
        |
        +---- Memory
        |
        v
Enterprise Services
        |
        v
PostgreSQL / S3 / Vector DB

---

## Model Serving

Development:

FastAPI
   |
   v
Ollama
   |
   v
Open-weight LLM


Production:

FastAPI
   |
   v
vLLM API
   |
   v
AWS GPU
   |
   v
Open-weight LLM