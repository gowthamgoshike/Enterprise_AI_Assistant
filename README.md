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
