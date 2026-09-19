# PostgreSQL conversation checkpoints

Install dependencies with `.venv/bin/pip install -r requirements.txt` and set
`LANGGRAPH_DATABASE_URL=postgresql://user:password@localhost:5432/enterprise_ai`
in `.env`, using your actual credentials. This is a psycopg connection URL;
do not use the SQLAlchemy `postgresql+psycopg2://` prefix.

FastAPI's lifespan opens an AsyncPostgresSaver, runs setup, and creates one
graph in `app.state.enterprise_graph`. Shutdown closes the connection.
The configured database must already exist; setup creates checkpoint tables.
Missing configuration or connection/setup failures prevent startup.

Both chat endpoints submit only the newest message with the session ID as the
LangGraph thread ID. Earlier messages and summaries are restored by PostgreSQL.
The UI keeps history for display but sends only the latest user message.
Existing in-memory sessions are not migrated.

To check persistence, send a message identifying a transaction, restart the API,
and send a follow-up with the same session ID and only the new message. The
earlier conversation should still be available. A different session ID should
have separate history. Clearing the UI creates a new session; it does not delete
existing PostgreSQL checkpoints.

Unit tests and standalone examples explicitly supply InMemorySaver. Run unit
tests with `.venv/bin/python -m pytest tests/unit -q`.

Reference: https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver
