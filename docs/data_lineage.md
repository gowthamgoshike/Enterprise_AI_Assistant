# Transaction Data Lineage

## Example

transaction_id:
TX-abc123

event_id:
EVT-xyz789

## Flow

Producer
→ Kafka topic: transactions
→ Partition: 1
→ Offset: 42
→ Bronze transaction dataset
→ Silver clean transaction dataset
→ PostgreSQL transactions table

## Trace identifiers

- event_id
- transaction_id
- kafka_partition
- kafka_offset
- event_time