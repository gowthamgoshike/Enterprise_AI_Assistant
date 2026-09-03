# Spark Structured Streaming

## Source

Kafka topic:

transactions

## Processing

1. Read Kafka messages
2. Decode binary key/value
3. Parse transaction JSON
4. Apply schema
5. Validate amounts/currency
6. Add processing timestamp
7. Deduplicate by event_id
8. Track event-time watermark

## Current Sink

Console

## Future Sink

AWS S3 Bronze layer