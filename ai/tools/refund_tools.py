from langchain_core.tools import tool


@tool
def propose_refund(
    transaction_id: str,
    reason: str,
) -> dict:
    """
    Propose a refund for a completed transaction.

    This does not execute a refund. The proposal uses the transaction's trusted
    amount and currency and requires explicit human approval before any future
    transaction modification.
    """
    raise RuntimeError("propose_refund requires trusted execution context")
