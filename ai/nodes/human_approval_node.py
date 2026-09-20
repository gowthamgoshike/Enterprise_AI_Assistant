import logging
from typing import Literal

from langgraph.types import Command, interrupt

from ai.schemas.actions import ApprovalDecision
from ai.schemas.graph_state import EnterpriseAgentState
from core.observability.metrics import APPROVAL_DECISIONS_TOTAL


logger = logging.getLogger(__name__)


def human_approval_node(
    state: EnterpriseAgentState,
) -> Command[Literal["execute_refund", "action_rejected"]]:
    """Pause the graph and route only after a validated human decision."""
    decision = interrupt({
        "type": "action_approval",
        "title": "Refund approval required",
        "action": state["pending_action"],
        "message": "Review this refund before continuing.",
    })
    approval = ApprovalDecision.model_validate(decision)
    approval_result = "APPROVED" if approval.approved else "REJECTED"
    proposal = state["pending_action"]
    APPROVAL_DECISIONS_TOTAL.labels(
        action=proposal["action_type"],
        decision=approval_result.casefold(),
    ).inc()
    logger.info(
        "human_approval_resolved request_id=%s user_id=%s "
        "action_id=%s action_type=%s approval=%s",
        state["request_id"],
        state["user_id"],
        proposal["action_id"],
        proposal["action_type"],
        approval_result,
        extra={
            "event": "approval_decision",
            "request_id": state["request_id"],
            "user_id": state["user_id"],
            "action_id": proposal["action_id"],
            "action_type": proposal["action_type"],
            "approval": approval_result,
            "decision": approval_result.casefold(),
        },
    )

    if approval.approved:
        return Command(
            update={"approval_status": "approved"},
            goto="execute_refund",
        )
    return Command(
        update={"approval_status": "rejected"},
        goto="action_rejected",
    )
