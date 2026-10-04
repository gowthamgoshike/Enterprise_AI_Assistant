# Enterprise Real-Time Transaction AI Assistant

An enterprise AI assistant for transaction lookup, revenue analytics, and approval-based refunds. It combines a Streamlit interface, a FastAPI backend, and a LangGraph tool-calling agent with a Kafka and Apache Spark transaction pipeline.

The model selects approved tools; business services enforce authorization and perform database operations. PostgreSQL stores operational data and conversation checkpoints, while Redis supports caching and request limits.

## Features

- **Authenticated conversations:** JWT login, Argon2 password hashing, and role-based access for customers, analysts, and administrators.
- **Transaction tools:** look up transactions and retrieve customer-scoped transaction history.
- **Analytics:** query revenue and transaction metrics, with Redis caching and background report jobs.
- **Human approval:** propose refunds and require an explicit approval step before execution, with audit records.
- **Persistent memory:** retain conversation state in PostgreSQL through LangGraph checkpoints; support streaming chat and context summarization.
- **Model providers:** use Ollama locally or connect to an OpenAI-compatible vLLM server.
- **Streaming data:** ingest Kafka events into PostgreSQL/RDS or a Bronze/Silver/Gold Parquet data lake on local storage or S3.
- **Reliability and evaluation:** bounded retries, timeouts, concurrency limits, health endpoints, structured logs, Prometheus metrics, and behavioral evaluation suites.

This is an evolving engineering project. The RAG package is currently a placeholder; policy-document retrieval, AWS GPU deployment, and CI/CD are future work rather than a complete deployment included here.

## Architecture

```mermaid
flowchart TD
    User --> UI[Streamlit UI]
    UI --> API[FastAPI: authentication and authorization]
    API --> Agent[LangGraph agent]
    Agent <--> Model[Ollama or vLLM]
    Agent --> Tools[Approved tools and business services]
    Tools --> DB[(PostgreSQL / RDS)]
    Agent <--> Memory[(PostgreSQL checkpoints)]
    Tools <--> Redis[(Redis cache)]
    API --> Redis

    Producer[Transaction producer] --> Kafka
    Kafka --> Stream[Spark transaction stream]
    Stream --> DB
    Kafka --> Bronze[Bronze Parquet]
    Bronze --> Silver[Silver Parquet]
    Silver --> Gold[Gold Parquet]
    Silver --> Loaders[Batch loaders]
    Gold --> Loaders
    Loaders --> DB
```

The direct PostgreSQL stream and the data-lake jobs are separate execution paths. Spark jobs and loaders must be started explicitly; the application Compose stack does not orchestrate them.

## Repository layout

| Path | Purpose |
| --- | --- |
| `apps/api/` | FastAPI routes, middleware, schemas, and chat services |
| `apps/ui/` | Streamlit interface and API clients |
| `ai/` | Agents, graphs, tools, model providers, memory, and evaluations |
| `core/` | Configuration, database pools, Redis, retries, and observability |
| `services/` | Authentication, transactions, refunds, analytics, and jobs |
| `repositories/` | Background-job persistence |
| `streaming/` | Kafka producers/consumers and Spark processing jobs |
| `data/migrations/` | Ordered SQL schema migrations |
| `infrastructure/docker/` | API/UI Dockerfiles and Compose configuration |
| `infrastructure/terraform/` | AWS network, private RDS, S3 data lake, IAM, and infrastructure tests |
| `tests/` | Unit, agent, and integration tests |
| `docs/` | Architecture, data contracts, and design notes |

## Local setup

Run commands from the repository root. Use Python **3.11**, matching the Docker images. You also need PostgreSQL, Ollama with a tool-capable model installed, and Docker Compose for Redis. Kafka and a Java/Spark environment are only needed for data ingestion.

### 1. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.runtime.txt
```

For development, tests, and streaming jobs, install the full dependency set instead:

```bash
python -m pip install -r requirements.txt
```

### 2. Configure the application

Copy the example only if you do not already have a `.env` file:

```bash
cp -n .env.example .env
```

Edit `.env` with your database credentials and installed model name. For an API running directly on your machine, use:

```dotenv
APP_ENV=development
MODEL_PROVIDER=ollama
MODEL_NAME=your-installed-model-name
MODEL_BASE_URL=http://localhost:11434
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=enterprise_ai
DATABASE_USER=enterprise_user
DATABASE_PASSWORD=your-database-password
DATABASE_URL=postgresql+psycopg2://enterprise_user:your-database-password@localhost:5432/enterprise_ai
ASYNC_DATABASE_URL=postgresql+asyncpg://enterprise_user:your-database-password@localhost:5432/enterprise_ai
LANGGRAPH_DATABASE_URL=postgresql://enterprise_user:your-database-password@localhost:5432/enterprise_ai
REDIS_URL=redis://localhost:6379/0
API_BASE_URL=http://localhost:8000
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

Generate a JWT secret with `python -c 'import secrets; print(secrets.token_urlsafe(48))'`. Percent-encode special characters in credentials embedded in URLs. Keep `.env` out of version control.

`DATABASE_URL` takes precedence over individual `DATABASE_*` fields. If you remove `DATABASE_URL` and `ASYNC_DATABASE_URL`, the application derives them from the individual fields. Keep the checkpoint URL aligned with the intended database; it uses a plain `postgresql://` URL, without a SQLAlchemy driver suffix.

Check that Ollama is running and that `MODEL_NAME` appears in the installed models:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### 3. Prepare PostgreSQL and a login

Create the configured PostgreSQL database and role using your database administration tools, or reuse an existing project database. Compose does **not** create PostgreSQL or seed accounts.

For a new database, apply the SQL files in `data/migrations/` in numeric order. With the local values above and `psql` installed:

```bash
for migration in data/migrations/*.sql; do
  psql -h localhost -p 5432 -U enterprise_user -d enterprise_ai \
    -v ON_ERROR_STOP=1 -f "$migration" || break
done
```

Resolve any migration error before continuing. For an existing database, apply only outstanding migrations.

Generate a password hash interactively:

```bash
python scripts/generate_password.py
```

Insert a development account through your SQL client, replacing the hash placeholder with the generated value:

```sql
INSERT INTO users (user_id, email, password_hash, role, customer_id)
VALUES ('USER-DEV-001', 'analyst@example.com', '<generated-argon2-hash>', 'ANALYST', NULL);
```

Supported roles are `CUSTOMER`, `ANALYST`, and `ADMIN`. Customers need a `customer_id` matching their transaction records. Analysts can read transaction and analytics data and generate reports; administrators additionally have refund permissions. There is no public registration endpoint.

For the first startup against a new checkpoint database, set `INITIALIZE_CHECKPOINT_SCHEMA=true` in `.env`. This allows the API to create LangGraph checkpoint tables. After successful initialization, set it back to `false` and restart the API. The default is `false`, so an uninitialized database needs this explicit setup step.

### 4. Start Redis, the API, and the UI

```bash
docker compose --env-file .env -f infrastructure/docker/compose.yaml up -d redis
python -m uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, activate the virtual environment and start the UI:

```bash
source .venv/bin/activate
python -m streamlit run apps/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

Open the [UI](http://localhost:8501) and log in with the account you created. The [API documentation](http://localhost:8000/docs) is available outside production mode.

```bash
curl http://localhost:8000/api/v1/health/ready
```

An empty database can support login and chat once initialized, but meaningful transaction and analytics answers require ingested or loaded data.

## Run the application with Docker Compose

Compose starts **Redis, FastAPI, and Streamlit**. PostgreSQL and model serving remain external dependencies. Complete the database and account setup above first.

For a database and Ollama running on the Docker host, adjust `.env`:

```dotenv
DATABASE_HOST=host.docker.internal
LANGGRAPH_DATABASE_URL=postgresql://enterprise_user:your-database-password@host.docker.internal:5432/enterprise_ai
MODEL_BASE_URL=http://host.docker.internal:11434
```

For RDS or another remote database, use its reachable hostname and appropriate SSL settings instead. The Compose API builds its database URL from the individual `DATABASE_*` fields; it does not pass through the root `.env` file's `DATABASE_URL` or `ASYNC_DATABASE_URL`. It sets Redis to `redis://redis:6379/0`, and the UI connects to `http://api:8000`.

```bash
docker compose --env-file .env -f infrastructure/docker/compose.yaml up --build -d
docker compose --env-file .env -f infrastructure/docker/compose.yaml ps
curl http://localhost:8000/api/v1/health/ready
```

Use the same UI and API URLs as the local setup. Stop any host-run API/UI first to avoid port conflicts. Ollama must listen on an interface reachable from the API container.

```bash
# View application logs
docker compose --env-file .env -f infrastructure/docker/compose.yaml logs -f api

# Stop the application stack
docker compose --env-file .env -f infrastructure/docker/compose.yaml down
```

This does not stop your separately managed PostgreSQL instance.

## Using the assistant

Try requests appropriate to the signed-in user's role and the records in your database:

- Customer: “Show my recent transactions.”
- Analyst: “What is the status of transaction TX-1001?”
- Analyst: “Show revenue by product.”
- Administrator: “Propose a refund for transaction TX-1001.”

Refund requests may return `approval_required`; submit the approval or rejection through the UI before the workflow continues. Keep the same session for follow-up questions. Starting a new UI conversation creates a new session; it does not delete stored checkpoints.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/auth/login` | Exchange email/password for a bearer token |
| `GET /api/v1/auth/me` | Inspect the authenticated user |
| `POST /api/v1/chat` | Send messages with a session ID |
| `POST /api/v1/chat/stream` | Receive server-sent chat events |
| `POST /api/v1/chat/approval` | Approve or reject a pending action |
| `POST /api/v1/jobs` | Submit a background job |
| `GET /api/v1/jobs/{job_id}` | Read job status |
| `POST /api/v1/jobs/{job_id}/cancel` | Cancel a job |
| `GET /api/v1/health/live` | Check process liveness |
| `GET /api/v1/health/ready` | Check dependency readiness |
| `GET /metrics` | Read Prometheus metrics |

See `/docs` for request schemas and authorization requirements.

## Transaction ingestion and the data lake

Install `requirements.txt` and provide a running Kafka broker with the `transactions` topic. Set `KAFKA_BOOTSTRAP_SERVERS` and `KAFKA_TOPIC` in `.env` for the producer and direct PostgreSQL stream.

Start the synthetic transaction producer:

```bash
python -m streaming.producer.continuous_kafka_producer
```

In another terminal, submit the direct database stream using the connector matching the pinned Spark 3.5.8 dependency:

```bash
python -m dotenv run -- spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 \
  streaming/spark/transaction_stream.py
```

The stream validates positive amounts and supported currencies, deduplicates events, and writes micro-batches to PostgreSQL/RDS. `RDS_STREAM_CHECKPOINT_LOCATION` controls recovery state. `SPARK_DATABASE_HOST` and `SPARK_DATABASE_PORT` can override database routing when Spark runs outside the application's network.

For the separate data-lake path:

1. Run `streaming/spark/bronze_stream.py` with the same Kafka connector to write Bronze Parquet. This script currently uses `localhost:9092` and the `transactions` topic directly.
2. Run `streaming/spark/silver_transform.py` to transform Bronze data.
3. Run `streaming/spark/gold_transform.py` to create aggregates.
4. Load serving data with `python -m services.transaction_service.load_transactions` and, when needed, `python -m services.analytics_service.load_gold`.

The Gold loader replaces the `product_revenue` table contents. Run it only against the intended serving database.

`DATA_ROOT` defaults to `data_lake`; set it to an S3A URI for S3 storage. S3 jobs require Hadoop AWS support matching the installed Spark/Hadoop distribution, for example:

```bash
DATA_ROOT=s3a://your-bucket/enterprise-ai python -m dotenv run -- spark-submit \
  --packages org.apache.hadoop:hadoop-aws:3.3.4 \
  streaming/spark/silver_transform.py
```

Use an AWS profile or IAM role for credentials. The `dotenv run` prefix makes environment variables available before Spark starts its JVM. Run one Bronze writer per output path and preserve both its checkpoint and `_spark_metadata` directories for recovery.

## Configuration and operations

See [.env.example](.env.example) and [core/config.py](core/config.py) for the complete settings.

| Setting | Default / behavior |
| --- | --- |
| `MODEL_PROVIDER` | `ollama`; also supports `vllm` |
| `MODEL_BASE_URL` | Provider address; for vLLM, use the OpenAI-compatible API base such as `http://localhost:8001/v1` |
| `MODEL_API_KEY` | Optional model-server authentication |
| `MODEL_MAX_CONCURRENCY` | 2 concurrent model requests |
| `MAX_TOOL_CALLS_PER_REQUEST` | 5 tool calls |
| `REDIS_CACHE_TTL_SECONDS` | 30-second analytics cache |
| `RATE_LIMIT_REQUESTS` | 20 chat requests per user per window |
| `RATE_LIMIT_WINDOW_SECONDS` | 60 seconds |
| `INITIALIZE_CHECKPOINT_SCHEMA` | `false`; enable explicitly for checkpoint setup |
| `APP_ENV` | `development`, `testing`, `staging`, or `production` |

If Redis is unavailable, analytics falls back to PostgreSQL. Readiness can return HTTP 200 with a `degraded` status when core dependencies are healthy. Development chat rate limiting allows requests through during Redis outages. Login, jobs, and approvals have separately configurable limits.

For startup or readiness failures, check database connectivity, checkpoint initialization, and model availability. In Docker, `localhost` refers to the container itself; use a reachable external hostname for PostgreSQL and Ollama. A `401` response calls for checking account credentials or token expiry; a `429` response indicates a request limit was reached.

## Day 43 — Terraform infrastructure

The [Terraform guide](infrastructure/terraform/README.md) adds an AWS foundation: a VPC with private subnets, PostgreSQL RDS, a versioned S3 data lake, and a scoped EC2 instance profile. It includes example variables, optional remote-state configuration, and tests using a mocked AWS provider.

```bash
terraform -chdir=infrastructure/terraform init -backend=false
terraform -chdir=infrastructure/terraform fmt -check -recursive
terraform -chdir=infrastructure/terraform validate
terraform -chdir=infrastructure/terraform test
```

These checks do not provision AWS resources. Follow the guide to configure, review a plan, and deliberately apply it. Application compute and private connectivity are separate steps; this foundation does not automatically migrate an existing database or deploy the app.

## Tests and evaluations

Install the full dependencies and configure the required database and JWT settings before running tests:

```bash
python -m pytest tests/unit tests/agent -q
```

Integration tests live in `tests/integration/`; use a dedicated test database and inspect their fixtures before running them.

Behavioral evaluations check tool selection, grounding, authorization, conversation isolation, and approval workflows. They use the configured model and data services:

```bash
python -m ai.evaluations.runner --suite smoke
```

For fixture-backed evaluations, point all database URLs at a disposable test or staging database:

```bash
ALLOW_EVALUATION_FIXTURES=true APP_ENV=testing \
  python -m ai.evaluations.runner --suite full --load-fixtures
```

The explicitly selected `write` suite exercises a real refund on disposable evaluation data:

```bash
ALLOW_EVALUATION_FIXTURES=true APP_ENV=testing \
  python -m ai.evaluations.runner --suite write --load-fixtures
```

Reports are written under `evaluation_results/`. Fixture loading requires explicit opt-in and refuses production environments. Fault-injection cases are a separate suite intended for controlled failure testing.

## Further documentation

- [Architecture overview](docs/architecture.md)
- [LangGraph architecture](docs/langgraph_architecture.md)
- [LangChain architecture](docs/langchain_architecture.md)
- [Tool design](docs/tools.md)
- [Transaction event schema](docs/transaction_event_schema.md)
- [Data sources](docs/data_sources.md) and [data lineage](docs/data_lineage.md)
- [Frontend design](docs/frontend.md)
- [Database access decision](docs/adr/ADR-001-ai-database-access.md)

Some design notes describe earlier stages of the project; the current source and configuration define runtime behavior.
