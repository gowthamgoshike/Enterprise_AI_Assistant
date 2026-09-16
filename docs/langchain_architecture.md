# LangChain Architecture

## Purpose

LangChain is used for AI orchestration. It does not replace business services, repositories, authorization, or the database.

## Flow

User
-> FastAPI
-> LangChain agent service
-> ChatOllama
-> Tool decision
-> LangChain Tool
-> Business Service
-> Repository
-> PostgreSQL
-> ToolMessage
-> LLM
-> Response

## Current Tool

`get_transaction` retrieves the current state of a transaction by transaction ID.

The tool is explicitly allowlisted, and its arguments are validated with `GetTransactionArguments` before execution. Tool results are serialized as JSON and restricted to the fields needed by the response.

## Services

The existing manual agent remains in `apps/api/services/agent_service.py` for comparison. The LangChain implementation is in `apps/api/services/langchain_agent_service.py` and currently performs one tool round with an asynchronous model invocation.

A maximum-step constant is present as the loop expands. Multiple tool calls and multiple rounds should be handled explicitly before this service is used as a production agent.

## Security

- Tools are explicitly bound and allowlisted.
- Tool arguments are validated before execution.
- The model has no database credentials.
- Arbitrary SQL is not exposed.
- Business services remain the trusted boundary.
- Tool results are sanitized before being sent back to the model.
- Missing transactions are returned as `found: false` so the model can state that no transaction was found.
