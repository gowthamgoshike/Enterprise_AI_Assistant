# Enterprise AI Tools

## get_transaction

Purpose:

Retrieve the current serving-state record
for a specific transaction.

Input:

transaction_id

Example:

TX-123

Data flow:

AI Agent
→ Transaction Tool
→ Transaction Service
→ Repository
→ PostgreSQL

Security principles:

- Tool must be allowlisted.
- Inputs must be validated.
- SQL must be parameterized.
- The LLM does not receive DB credentials.
- The LLM does not generate arbitrary SQL.
- Output should contain only required fields.