from services.transaction_service.service import (
    find_transaction,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
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

def get_transaction_tool(
    transaction_id: str,
) -> dict:

    args = GetTransactionArguments(
        transaction_id=
            transaction_id
    )

    return find_transaction(
        args.transaction_id
    )

