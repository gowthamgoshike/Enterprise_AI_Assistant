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

## Run the API and UI with Docker Compose

Install Docker Desktop and Ollama on the Mac. Run `ollama list` and
`curl http://localhost:11434/api/tags` first. Set `OLLAMA_MODEL` in the
root `.env` to an installed model. The `DATABASE_USER`,
`DATABASE_PASSWORD`, `DATABASE_NAME`, and `DATABASE_PORT` values in `.env`
must match the existing Postgres container. The API also requires
`JWT_SECRET_KEY`.

From the repository root:

```bash
docker start enterprise-postgres  # if it is stopped
docker compose --env-file .env -f infrastructure/docker/compose.yaml up --build -d
docker compose --env-file .env -f infrastructure/docker/compose.yaml ps
curl http://localhost:8000/api/v1/health/ready
```

Open the UI at `http://localhost:8501`. The API is at
`http://localhost:8000`. Compose starts Redis, the API, and the UI. The API
reaches the already running Postgres container through
`host.docker.internal:${DATABASE_PORT}` and uses its existing users,
transactions, and checkpoints. It reaches Redis at `redis:6379` and Ollama
at `host.docker.internal:11434`. The UI reaches the API at `http://api:8000`.
Aggregate analytics results are cached for 30 seconds by default. If Redis is
unavailable, analytics queries use PostgreSQL directly; readiness reports
`degraded` but returns HTTP 200 when the core dependencies are healthy.
Chat and streaming chat share a per-user Redis limit of 20 requests per 60
seconds by default. Exceeding it returns HTTP 429; if Redis is unavailable,
development requests continue. Set `RATE_LIMIT_REQUESTS` and
`RATE_LIMIT_WINDOW_SECONDS` in `.env` to change the limit. Analytics cache
keys start with `v1:analytics:`; chat limit keys start with `v1:rate:chat:`.
No database initialization or account migration is needed. The images
install `requirements.runtime.txt`, which omits the Spark, Kafka, and test
packages used outside the API and UI.

To check model access from the API container:

```bash
docker compose --env-file .env -f infrastructure/docker/compose.yaml exec api \
  python -c "import httpx; print(httpx.get('http://host.docker.internal:11434/api/tags').status_code)"
```

If that fails while the host curl works, configure Ollama to listen on an
interface reachable from Docker Desktop without exposing it to the public
network. To stop the Compose services without stopping the existing database, run:

```bash
docker compose --env-file .env -f infrastructure/docker/compose.yaml down
```

The earlier `docker_postgres_data` volume from the separate Compose database
is not used by this setup and has not been deleted.

## Run Ollama, API, and UI on the Mac

Ollama already runs outside Docker. To use only the model, run
`ollama run qwen3.5:4b` (or replace the name with another installed model).
No project files need to change for that.

To run the complete app on the Mac while keeping the existing
`enterprise-postgres` container for data, start that container and stop the
Compose API/UI so ports 8000 and 8501 are free:

```bash
docker start enterprise-postgres
docker compose --env-file .env -f infrastructure/docker/compose.yaml down
docker compose --env-file .env -f infrastructure/docker/compose.yaml up -d redis
ollama list
curl http://localhost:11434/api/tags
.venv/bin/python -m pip install -r requirements.runtime.txt
```

The root `.env` should use `OLLAMA_BASE_URL=http://localhost:11434`, an
installed `OLLAMA_MODEL`, `API_BASE_URL=http://localhost:8000`, and the
existing database's `DATABASE_HOST=localhost`, `DATABASE_PORT=5432`,
`DATABASE_NAME`, `DATABASE_USER`, and `DATABASE_PASSWORD`. Set
`LANGGRAPH_DATABASE_URL` to the same database at `localhost:5432` and keep
`JWT_SECRET_KEY` set. If you define `DATABASE_URL`, it takes precedence over
the individual `DATABASE_*` settings and must also point to the host
database. Set `REDIS_URL=redis://localhost:6379/0` for the host-run API.

In one terminal, from the repository root:

```bash
.venv/bin/python -m uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
.venv/bin/python -m streamlit run apps/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

Check `http://localhost:8000/api/v1/health/ready` and open
`http://localhost:8501`. These commands were also verified on spare ports
8001 and 8502 while the Compose API/UI were running.

To stop using Docker entirely, run PostgreSQL on the Mac or another host,
restore the existing database there, and point `DATABASE_*`,
`LANGGRAPH_DATABASE_URL`, and any `DATABASE_URL` at that PostgreSQL instance.
The Python application and Ollama need no code changes.

## Spark Data Lake Paths

The Spark Bronze, Silver, and Gold jobs read their paths from `DATA_ROOT`.
Set it to `data_lake` for local development or to an S3A URI for AWS:

```text
DATA_ROOT=data_lake
# DATA_ROOT=s3a://enterprise-ai-data
```

Spark jobs should be submitted with Hadoop AWS support matching the Hadoop
version bundled with the Spark distribution. This project currently uses
Hadoop `3.3.4`:

```bash
.venv/bin/python -m dotenv run -- spark-submit \
	--packages org.apache.hadoop:hadoop-aws:3.3.4 \
	streaming/spark/silver_transform.py
```

The `dotenv run --` prefix is required when using `.env` locally because
`spark-submit` starts the Spark JVM before the Python job imports
`load_dotenv()`. It makes the AWS variables available to the S3A filesystem.

Bronze checkpoints default to
`<DATA_ROOT>/_checkpoints/bronze_transactions`, so S3 deployments keep the
checkpoint beside the data. Run only one Bronze writer for a given output path,
and do not manually delete the output's `_spark_metadata` directory or its
checkpoint. Those two locations are the stream's recovery state.

Use the AWS SDK credential chain, such as an AWS CLI profile locally or an
IAM role on AWS. Do not put access keys in source code or `.env` files.

---

## Core Architecture Principle

Enterprise systems are the source of truth.

The LLM does not directly control the database.

The LLM reasons about user requests and selects approved tools.
Business services enforce authorization, validation, and data access.

## Agent evaluations

The golden behavioral benchmark lives in `ai/evaluations`. It validates tool
selection and arguments, no-tool behavior, grounding, authorization, approval
workflows, and multi-turn isolation without exact-matching natural-language
answers.

Run the fast critical set:

```bash
.venv/bin/python -m ai.evaluations.runner --suite smoke
```

Run all 40 standard cases against a disposable test or staging database:

```bash
ALLOW_EVALUATION_FIXTURES=true APP_ENV=test \
  .venv/bin/python -m ai.evaluations.runner \
  --suite full --load-fixtures
```

The fixture loader refuses production environments and requires the explicit
`ALLOW_EVALUATION_FIXTURES=true` opt-in. Results are written as versioned JSON
under `evaluation_results/`. Security and approval-workflow scores are hard
release gates; a single failure blocks release. The three outage cases are a
separate `fault-injection` suite and use the runner's fixture hook so tests can
mock database, model, and tool failures safely.

Approved and rejected write-path evaluations are isolated in the explicitly
selected `write` suite because the approved case performs a real disposable
test-data refund:

```bash
ALLOW_EVALUATION_FIXTURES=true APP_ENV=test \
  .venv/bin/python -m ai.evaluations.runner \
  --suite write --load-fixtures
```

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
