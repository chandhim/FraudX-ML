from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "fraud_detection.py")


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert not at.warning


def test_predict_shows_report():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "Predict")
    assert not at.exception
    assert len(at.dataframe) == 1
    assert [s.value for s in at.success][0] == "LEGITIMATE"


def test_reset_clears_inputs_and_results():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "Predict")
    click(at, "Reset")
    assert not at.exception
    assert at.number_input(key="amount").value == 0.0
    assert at.selectbox(key="transaction_type").value == "PAYMENT"
    assert len(at.dataframe) == 0


def test_zero_amount_is_blocked():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "Reset")
    click(at, "Predict")
    assert "Transaction amount must be greater than zero." in [e.value for e in at.error]
    assert len(at.dataframe) == 0
