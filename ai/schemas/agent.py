from services.transaction_service.identifiers import normalize_transaction_id

from typing import (
    Annotated,
    Literal,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    TypeAdapter,
)


class IntentRoute(BaseModel):
    """Classification with an uncalibrated model-generated routing signal."""

    route: Literal["transaction", "analytics", "general"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=250)


class GetTransactionArguments(
    BaseModel
):

    transaction_id: str = Field(
        min_length=1,
        max_length=100,
    )

    @field_validator(
        "transaction_id"
    )
    @classmethod
    def normalize_transaction_id(
        cls,
        transaction_id: str,
    ) -> str:

        normalized_id = normalize_transaction_id(transaction_id)

        if not normalized_id:

            raise ValueError(
                "transaction_id must not be blank"
            )

        return normalized_id

class RespondDecision(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    action: Literal["respond"]


class TransactionToolDecision(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    action: Literal["tool"]

    tool_name: Literal[
        "get_transaction"
    ]

    arguments: GetTransactionArguments


AgentDecision = Annotated[
    RespondDecision
    |
    TransactionToolDecision,
    Field(
        discriminator="action"
    ),
]

agent_decision_adapter = TypeAdapter(
    AgentDecision
)

class AgentResponse(BaseModel):

    response: str

    tool_used: str | None = None

    success: bool = True


class AgentError(BaseModel):

    error_code: str

    message: str

    retryable: bool = False


class RouterDecision(BaseModel):

    action: Literal[
        "respond",
        "get_transaction",
    ]

    transaction_id: (
        str | None
    ) = Field(
        default=None,
        max_length=100,
    )
