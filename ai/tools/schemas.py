GET_TRANSACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_transaction",
        "description": (
            "Retrieve the current state of "
            "an enterprise transaction using "
            "its transaction ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": (
                        "The transaction identifier, "
                        "for example TX-123."
                    ),
                },
            },
            "required": [
                "transaction_id",
            ],
        },
    },
}