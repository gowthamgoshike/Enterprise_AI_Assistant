from core.errors import (
    AppError,
    RefundNotAllowedError as CoreRefundNotAllowedError,
    TransactionNotFoundError as CoreTransactionNotFoundError,
)


class RefundError(AppError):
    pass


class TransactionNotFoundError(CoreTransactionNotFoundError, RefundError):
    pass


class RefundNotAllowedError(CoreRefundNotAllowedError, RefundError):
    pass
