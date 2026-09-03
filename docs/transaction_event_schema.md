# Transaction Event Schema

## Version

1.0

## Purpose

Defines the canonical transaction event exchanged by the
real-time enterprise transaction platform.

## Fields

event_id:
Unique event identifier used for tracing and deduplication.

transaction_id:
Business identifier for the transaction.

customer_id:
Customer associated with the transaction.

product_id:
Product associated with the transaction.

amount_cents:
Transaction amount represented as integer cents.

currency:
ISO-style currency identifier.

status:
Current transaction event state.

transaction_type:
Business operation represented by the event.

source:
Channel where transaction originated.

event_time:
Timestamp when the business event occurred.

ingestion_time:
Timestamp when the platform received/generated the event.

schema_version:
Schema version used for compatibility and evolution.