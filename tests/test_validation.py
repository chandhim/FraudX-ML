from validation import validate_transaction


def test_consistent_payment_has_no_issues():
    assert validate_transaction("PAYMENT", 1000, 10000, 9000, 0, 0) == ([], [])


def test_float_rounding_is_tolerated():
    assert validate_transaction("PAYMENT", 1000.1, 10000.3, 9000.2, 0, 0) == ([], [])


def test_zero_amount_is_error():
    errors, _ = validate_transaction("PAYMENT", 0, 0, 0, 0, 0)
    assert errors


def test_negative_value_is_error():
    errors, _ = validate_transaction("PAYMENT", 100, -1, 0, 0, 0)
    assert "Negative values are not allowed." in errors


def test_cash_in_increases_origin():
    assert validate_transaction("CASH_IN", 500, 1000, 1500, 0, 0) == ([], [])


def test_inconsistent_transfer_warns_but_does_not_block():
    errors, warnings = validate_transaction("TRANSFER", 181, 0, 0, 0, 181)
    assert not errors
    assert len(warnings) == 2


def test_large_transfer_is_flagged_high_risk():
    _, warnings = validate_transaction("TRANSFER", 60000, 60000, 0, 0, 60000)
    assert any("High-risk" in w for w in warnings)
