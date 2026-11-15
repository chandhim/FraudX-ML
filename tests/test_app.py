from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "fraud_detection.py")


def new_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def click(at, label=None, key=None):
    button = at.button(key) if key else next(b for b in at.button if b.label == label)
    button.click().run()


def verdict_html(at):
    return " ".join(m.value for m in at.markdown if "fx-verdict" in m.value and "<div" in m.value)


def test_app_loads_without_errors():
    at = new_app()
    assert not at.exception
    assert not at.warning
    assert "Pick a sample transaction" in at.info[0].value


@pytest.mark.parametrize(
    "preset, expected",
    [
        ("everyday_payment", "Looks legitimate"),
        ("cash_deposit", "Looks legitimate"),
        ("drained_transfer", "Likely fraud"),
        ("drain_and_cash_out", "Likely fraud"),
    ],
)
def test_presets_load_and_show_expected_verdict(preset, expected):
    at = new_app()
    click(at, key=f"preset_{preset}")
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
    click(at, key="preset_drained_transfer")
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
