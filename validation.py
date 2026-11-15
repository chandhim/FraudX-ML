"""Validation rules for transactions entered in the FraudX app."""
import math

TOLERANCE = 0.01  # one cent


def _same(a, b):
    return math.isclose(a, b, abs_tol=TOLERANCE)


def validate_transaction(transaction_type, amount, oldbalanceOrg, newBalanceOrig,
                         oldBalanceDest, newBalanceDest):
    """
    Validates transaction data.
    Errors block prediction (input cannot be scored at all).
    Warnings are shown but the transaction is still scored, because balance
    inconsistencies are themselves a fraud signal in the training data.
    Returns: errors (list), warnings (list)
    """
    errors = []
    warnings = []

    # ❌ ERRORS - STOP EXECUTION
    if min(amount, oldbalanceOrg, newBalanceOrig, oldBalanceDest, newBalanceDest) < 0:
        errors.append("Negative values are not allowed.")
    if amount <= 0:
        errors.append("Transaction amount must be greater than zero.")
    if errors:
        return errors, warnings

    origin_change = oldbalanceOrg - newBalanceOrig  # positive = money left origin
    dest_change = newBalanceDest - oldBalanceDest   # positive = money arrived at destination

    # ⚠️ WARNINGS - ALERT BUT CONTINUE
    expected_origin_change = -amount if transaction_type == "CASH_IN" else amount
    if not _same(origin_change, expected_origin_change):
        warnings.append(
            f"Origin balance change ({origin_change:,.2f}) doesn't match the expected "
            f"{expected_origin_change:,.2f} for a {transaction_type}."
        )

    if dest_change > TOLERANCE and not _same(dest_change, amount):
        warnings.append(
            f"Destination balance change ({dest_change:,.2f}) doesn't match "
            f"the transaction amount ({amount:,.2f})."
        )

    if transaction_type != "CASH_IN" and origin_change <= TOLERANCE and dest_change > TOLERANCE:
        warnings.append("Destination was credited but the origin account was not debited.")

    if transaction_type in ("TRANSFER", "CASH_OUT") and amount > 50_000:
        warnings.append(f"High-risk transaction: {transaction_type} of {amount:,.2f}.")

    return errors, warnings
