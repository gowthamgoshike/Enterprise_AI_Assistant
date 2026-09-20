class RefundError(Exception):
    pass


class TransactionNotFoundError(RefundError):
    pass


class RefundNotAllowedError(RefundError):
    pass
