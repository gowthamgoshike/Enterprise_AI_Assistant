# Streamlit Frontend

## Responsibilities

- User interface
- Display chat history
- Capture user input
- Call backend APIs
- Display system health
- Display controlled errors
- Maintain temporary local session state

## Non-Responsibilities

The frontend does NOT:

- Query PostgreSQL directly
- Call Ollama directly
- Enforce authorization
- Store database credentials
- Execute tools
- Contain business logic

## Architecture

User
→ Streamlit
→ FastAPI
→ AI Service
→ Ollama
