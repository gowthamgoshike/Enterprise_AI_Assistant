# LangGraph Architecture

## Purpose

LangGraph orchestrates the AI workflow. FastAPI calls
`run_enterprise_graph` as the primary AI entry point.

LangChain continues to provide:

- chat models
- messages
- tools
- structured outputs

## State

The graph carries:

- `user_message`
- `request_id`
- `route`
- `route_confidence`
- `response`
- `tools_used`
- `error`

`EnterpriseState` defines this state in `ai/schemas/graph_state.py`.
Nodes return updates that LangGraph merges into the state.

## Nodes

- `router`: classifies the request and records routing confidence.
- `transaction`: handles requests for transaction details.
- `analytics`: handles requests for business analytics.
- `general`: answers general questions.
- `clarify`: asks the user to clarify an uncertain request.

## Flow

```text
START → router → domain handler → END
```

After the router, confidence below `0.65` selects `clarify`. Otherwise,
the graph selects `transaction`, `analytics`, or `general`.
Each handler and the clarification node lead to `END`.
The state's `route` retains the router's classification even when the
graph selects the clarification node.

Transaction handler failures return the safe error code
`transaction_handler_failed` and a service-unavailable response.
An `error_node` function exists for future explicit error paths but is
not yet connected to the graph. Raw exceptions are not stored in state.

## Design Principle

Nodes orchestrate behavior.

Tools perform approved capabilities.

Services enforce business behavior.

Repositories access data.

## Code Responsibility Map

| Location | Responsibility |
| --- | --- |
| `apps/api/` | HTTP/API boundary |
| `ai/graphs/` | Workflow orchestration |
| `ai/agents/` | Specialized handlers |
| `ai/tools/` | Capabilities exposed to the LLM |
| `services/` | Business logic |
| `services/*/repository.py` | Repository layer for data access |
| PostgreSQL | Serving data |

The repository layer currently lives inside each service rather than in
a separate top-level `repositories/` directory.

Streamlit continues to call `POST /api/v1/chat`; the API contract keeps
the frontend independent of the orchestration implementation.
