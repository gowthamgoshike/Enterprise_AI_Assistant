# ADR-001: AI Database Access

## Status

Accepted

## Context

The enterprise AI assistant must answer questions using sensitive
transaction data.

Allowing an LLM unrestricted database access would create significant
security and reliability risks.

## Decision

The LLM will not receive unrestricted direct database access.

The LLM may only access enterprise data through approved application tools.

Architecture:

LLM
→ Tool
→ Business Service
→ Authorization
→ Database

## Consequences

Advantages:

- Stronger security
- Controlled queries
- Easier auditing
- Better authorization
- Reduced SQL injection risk
- Predictable application behavior

Tradeoff:

More backend code must be implemented compared with allowing the LLM
to generate arbitrary SQL.