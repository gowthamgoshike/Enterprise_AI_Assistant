"""Normalize generated UUID transaction IDs and legacy transaction IDs."""
from uuid import UUID


def normalize_transaction_id(transaction_id: str) -> str:
    value = transaction_id.strip()
    if not value:
        raise ValueError("transaction_id must not be blank")
    if value[:3].upper() == "TX-":
        try:
            return f"TX-{UUID(value[3:])}"
        except ValueError:
            pass
    return value.upper()
