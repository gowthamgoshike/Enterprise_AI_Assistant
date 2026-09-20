"""Low-cardinality Prometheus metrics for the application runtime."""

from prometheus_client import Counter, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
)
MODEL_REQUESTS_TOTAL = Counter(
    "model_requests_total",
    "Total model requests",
    ["model", "status"],
)
MODEL_LATENCY = Histogram(
    "model_request_duration_seconds",
    "Model request duration",
    ["model"],
)
TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool", "status"],
)
TOOL_LATENCY = Histogram(
    "tool_execution_duration_seconds",
    "Tool execution latency",
    ["tool"],
)
AGENT_ERRORS_TOTAL = Counter(
    "agent_errors_total",
    "Total agent errors",
    ["error_code"],
)
RETRIES_TOTAL = Counter(
    "operation_retries_total",
    "Total retry attempts",
    ["operation"],
)
APPROVAL_DECISIONS_TOTAL = Counter(
    "approval_decisions_total",
    "Human approval decisions",
    ["action", "decision"],
)
REFUND_EXECUTIONS_TOTAL = Counter(
    "refund_executions_total",
    "Refund execution results",
    ["result"],
)
GRAPH_NODE_EXECUTIONS = Counter(
    "graph_node_executions_total",
    "LangGraph node executions",
    ["node", "status"],
)
GRAPH_NODE_DURATION = Histogram(
    "graph_node_duration_seconds",
    "LangGraph node duration",
    ["node"],
)
AUTHORIZATION_DENIALS_TOTAL = Counter(
    "authorization_denials_total",
    "Authorization denials",
    ["role", "capability"],
)
