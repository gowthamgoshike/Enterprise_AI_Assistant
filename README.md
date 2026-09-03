# Enterprise Real-Time Transaction AI Assistant

## Project Overview

This project builds a production-oriented enterprise AI assistant
on top of a real-time transaction processing platform.

The system processes continuous transaction events using Kafka and
Apache Spark, stores historical and analytical data in AWS S3 using
the Medallion architecture, maintains operational/serving data in
PostgreSQL, and exposes enterprise data through an authenticated
agentic AI assistant.

The AI assistant uses an open-weight LLM locally through Ollama
during development and will later use vLLM on AWS GPU infrastructure
for production model serving.

---

## Primary Goal

The primary goal of this project is not simply to build a chatbot.

The goal is to learn and implement production AI Engineering concepts:

- LLM inference
- Agent architecture
- Tool calling
- RAG
- Memory
- Authentication
- Authorization
- Human-in-the-loop workflows
- AI evaluation
- Observability
- Model serving
- Docker
- AWS
- Terraform
- CI/CD

---

## Data Architecture

Transaction Producer
→ Kafka
→ Spark Structured Streaming
→ S3 Bronze
→ S3 Silver
→ S3 Gold
→ PostgreSQL

---

## AI Architecture

User
→ Streamlit
→ FastAPI
→ Authentication
→ Authorization
→ AI Agent
→ Tools / RAG / Memory
→ Enterprise Data Sources

Development LLM:
Ollama

Production LLM:
vLLM on AWS GPU infrastructure

---

## Core Architecture Principle

Enterprise systems are the source of truth.

The LLM does not directly control the database.

The LLM reasons about user requests and selects approved tools.
Business services enforce authorization, validation, and data access.

---

## Initial Use Cases

### UC01 — Transaction lookup

User:

"What's the status of transaction TX-1001?"

Data source:

PostgreSQL transaction tool.

---

### UC02 — Personal transaction history

User:

"Show my transactions from today."

Data source:

PostgreSQL with authenticated user filtering.

---

### UC03 — Revenue analytics

User:

"What was our revenue during the last hour?"

Data source:

Analytics service / PostgreSQL.

---

### UC04 — Failure analysis

User:

"Which products had unusually high transaction failure rates today?"

Data source:

Analytics tool.

---

### UC05 — Enterprise policy

User:

"What is our refund policy?"

Data source:

RAG knowledge base.