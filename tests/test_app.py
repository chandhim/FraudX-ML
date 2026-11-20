from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "fraud_detection.py")


def new_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def verdict_html(at):
    return " ".join(m.value for m in at.markdown if "fx-verdict" in m.value and "<div" in m.value)


def enter(at, transaction_type, amount, old_orig, new_orig, old_dest, new_dest):
    """Fill the form by hand, with the new balances typed in rather than auto-calculated."""
    at.toggle(key="auto_balances").set_value(False).run()
    at.selectbox(key="transaction_type").set_value(transaction_type)
    for key, value in zip(
        ["amount", "oldbalanceOrg", "newBalanceOrig", "oldBalanceDest", "newBalanceDest"],
        [amount, old_orig, new_orig, old_dest, new_dest],
    ):
        at.number_input(key=key).set_value(value)
    at.run()


def test_app_loads_without_errors():
    at = new_app()
    assert not at.exception
    assert not at.warning
    assert "Enter a transaction" in at.info[0].value


def test_no_sample_buttons():
    at = new_app()
    assert [b.label for b in at.button] == ["Analyze transaction", "Reset"]


@pytest.mark.parametrize(
    "transaction, expected",
    [
        (("PAYMENT", 4500.0, 32000.0, 27500.0, 0.0, 0.0), "Looks legitimate"),
        (("CASH_IN", 20000.0, 5000.0, 25000.0, 60000.0, 40000.0), "Looks legitimate"),
        (("TRANSFER", 181000.0, 181000.0, 0.0, 0.0, 0.0), "Likely fraud"),
        (("CASH_OUT", 350000.0, 350000.0, 0.0, 12000.0, 362000.0), "Likely fraud"),
    ],
)
def test_entered_transactions_show_expected_verdict(transaction, expected):
    at = new_app()
    enter(at, *transaction)
    click(at, "Analyze transaction")
    assert not at.exception
    assert expected in verdict_html(at)


def test_auto_balances_fill_new_balances():
    at = new_app()
    at.number_input(key="amount").set_value(2500.0)
    at.number_input(key="oldbalanceOrg").set_value(8000.0)
    at.selectbox(key="transaction_type").set_value("TRANSFER")
    at.number_input(key="oldBalanceDest").set_value(1000.0)
    at.run()
    assert at.number_input(key="newBalanceOrig").value == 5500.0
    assert at.number_input(key="newBalanceDest").value == 3500.0


def test_analyze_then_edit_marks_result_stale():
    at = new_app()
    click(at, "Analyze transaction")
    assert "Looks legitimate" in verdict_html(at)
    at.number_input(key="amount").set_value(5000.0).run()
    assert "inputs have changed" in at.info[0].value


def test_reset_clears_inputs_and_results():
    at = new_app()
    enter(at, "TRANSFER", 181000.0, 181000.0, 0.0, 0.0, 0.0)
    click(at, "Analyze transaction")
    assert "Likely fraud" in verdict_html(at)
    click(at, "Reset")
    assert not at.exception
    assert at.number_input(key="amount").value == 0.0
    assert at.selectbox(key="transaction_type").value == "PAYMENT"
    assert at.toggle(key="auto_balances").value is True
    assert verdict_html(at) == ""


def test_zero_amount_is_blocked():
    at = new_app()
    click(at, "Reset")
    click(at, "Analyze transaction")
    assert "Transaction amount must be greater than zero." in [e.value for e in at.error]
    assert verdict_html(at) == ""
