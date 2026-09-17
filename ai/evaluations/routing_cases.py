ROUTING_CASES = [
    {"question": "What is TX-100's status?", "expected": "transaction"},
    {"question": "How much was TX-00249c43-3574-40e9-8430-de137dd7581a?", "expected": "transaction"},
    {"question": "Compare TX-00249c43-3574-40e9-8430-de137dd7581a and TX-0025a6a2-94e3-41c8-b7c3-24aba137ba81.", "expected": "transaction"},
    {"question": "Show top 5 products.", "expected": "analytics"},
    {"question": "What is the revenue summary?", "expected": "analytics"},
    {"question": "How many transactions failed?", "expected": "analytics"},
    {"question": "Explain Kafka.", "expected": "general"},
    {"question": "What is Spark Streaming?", "expected": "general"},
]
