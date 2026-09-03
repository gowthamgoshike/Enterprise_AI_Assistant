
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from data.schemas.transaction import (
    Currency,
    TransactionEvent,
    TransactionSource,
    TransactionStatus,
    TransactionType,
)


def create_valid_transaction():

    now = datetime.now(timezone.utc)

    return TransactionEvent(
        event_id="EVT-1",
        transaction_id="TX-1",
        customer_id="C-1",
        product_id="P-1",
        amount_cents=2500,
        currency=Currency.USD,
        status=TransactionStatus.COMPLETED,
        transaction_type=TransactionType.PURCHASE,
        source=TransactionSource.WEB,
        event_time=now,
        ingestion_time=now,
    )


def test_valid_transaction():

    event = create_valid_transaction()

    assert event.transaction_id == "TX-1"

    assert event.amount_cents == 2500

    assert event.currency == Currency.USD


def test_negative_amount_rejected():

    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):

        TransactionEvent(
            event_id="EVT-1",
            transaction_id="TX-1",
            customer_id="C-1",
            product_id="P-1",
            amount_cents=-500,
            currency=Currency.USD,
            status=TransactionStatus.COMPLETED,
            transaction_type=TransactionType.PURCHASE,
            source=TransactionSource.WEB,
            event_time=now,
            ingestion_time=now,
        )


def test_invalid_status_rejected():

    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):

        TransactionEvent(
            event_id="EVT-1",
            transaction_id="TX-1",
            customer_id="C-1",
            product_id="P-1",
            amount_cents=5000,
            currency=Currency.USD,
            status="unknown-status",
            transaction_type=TransactionType.PURCHASE,
            source=TransactionSource.WEB,
            event_time=now,
            ingestion_time=now,
        )