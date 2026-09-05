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

        normalized_id = transaction_id.strip().upper()

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


