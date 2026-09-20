from enum import Enum


class UserRole(str, Enum):
    CUSTOMER = "CUSTOMER"
    ANALYST = "ANALYST"
    ADMIN = "ADMIN"
