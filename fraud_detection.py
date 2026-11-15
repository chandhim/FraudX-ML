import math
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_detection_model.pkl"
LOGO_PATH = BASE_DIR / "images" / "FraudX.svg"
# ===========================================================

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="FraudX",
    page_icon=str(LOGO_PATH),
    layout="wide"
)
# ===========================================================


@st.cache_resource
def load_model(path):
    return joblib.load(path)


model = load_model(MODEL_PATH)

# ==================== SESSION STATE INITIALIZATION ====================
TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

DEFAULTS = {
    "transaction_type": "PAYMENT",
    "amount": 1000.0,
    "oldbalanceOrg": 10000.0,
    "newBalanceOrig": 9000.0,
    "oldBalanceDest": 5000.0,
    "newBalanceDest": 0.0,
}
RESET_VALUES = {
    "transaction_type": "PAYMENT",
    "amount": 0.0,
    "oldbalanceOrg": 0.0,
    "newBalanceOrig": 0.0,
    "oldBalanceDest": 0.0,
    "newBalanceDest": 0.0,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_inputs():
    # Runs as an on_click callback, i.e. before widgets are re-created,
    # which is the only point where widget-backed state may be changed.
    for key, value in RESET_VALUES.items():
        st.session_state[key] = value
# ===========================================================

# ==================== VALIDATION LAYER ====================
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
        warnings.append(f"High-risk transaction: {transaction_type} of LKR {amount:,.2f}.")

    return errors, warnings
# ===========================================================

# ==================== HEADER WITH LOGO ====================
col1, col2 = st.columns([1, 4])

with col1:
    st.image(str(LOGO_PATH), width=200)

with col2:
    st.title("Advanced Fraud Detection System")
    st.markdown("Powered by Machine Learning to identify fraudulent transactions in real-time.")
st.divider()

st.markdown("### Please input the transaction details below")

transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES, key="transaction_type")
amount = st.number_input("Transaction Amount", min_value=0.0, key="amount")
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, key="oldbalanceOrg")
newBalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, key="newBalanceOrig")
oldBalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, key="oldBalanceDest")
newBalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, key="newBalanceDest")

col_predict, col_reset = st.columns(2)
predict_clicked = col_predict.button("Predict", type="primary", width="stretch")
col_reset.button("Reset", on_click=reset_inputs, width="stretch")

if predict_clicked:
    # Validation Layer
    errors, warnings = validate_transaction(
        transaction_type, amount, oldbalanceOrg, newBalanceOrig,
        oldBalanceDest, newBalanceDest
    )

    # Stop on Errors
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        st.stop()

    # Show Warnings
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")

    # Proceed with prediction (column names must match the training data)
    input_data = pd.DataFrame(
        [
            {
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newBalanceOrig,
                "oldbalanceDest": oldBalanceDest,
                "newbalanceDest": newBalanceDest,
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    # Get confidence score
    try:
        confidence_scores = model.predict_proba(input_data)[0]
        confidence = max(confidence_scores) * 100
        fraud_prob = confidence_scores[1] * 100 if len(confidence_scores) > 1 else 0
    except AttributeError:
        confidence = 100.0
        fraud_prob = 100.0 if prediction == 1 else 0.0

    st.divider()

    # ==================== RESULT VISUALIZATIONS ====================
    st.markdown("### Prediction Result")

    # Row 1: Main Prediction Status
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if prediction == 1:
            st.error("FRAUDULENT")
        else:
            st.success("LEGITIMATE")

    with col2:
        st.metric("Model Confidence", f"{confidence:.1f}%")

    with col3:
        st.metric("Fraud Probability", f"{fraud_prob:.1f}%")

    st.markdown("---")

    # Row 2: Transaction Summary
    st.markdown("### Transaction Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Transaction Type", transaction_type)

    with col2:
        st.metric("Amount", f"LKR {amount:,.2f}")

    with col3:
        st.metric("Origin Balance Change", f"LKR {newBalanceOrig - oldbalanceOrg:+,.2f}")

    with col4:
        st.metric("Destination Balance Change", f"LKR {newBalanceDest - oldBalanceDest:+,.2f}")

    st.markdown("---")

    # Row 3: Confidence Gauge and Risk Indicator
    st.markdown("### Risk Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Confidence Progress Bar
        st.markdown("**Model Confidence Level:**")
        st.progress(min(confidence / 100, 1.0))
        st.caption(f"{confidence:.1f}% confident in prediction")

    with col2:
        # Risk Gauge
        st.markdown("**Fraud Risk Level:**")
        if fraud_prob < 25:
            st.success("🟢 **LOW RISK**")
        elif fraud_prob < 50:
            st.info("🟡 **MEDIUM RISK**")
        elif fraud_prob < 75:
            st.warning("🟠 **HIGH RISK**")
        else:
            st.error("🔴 **CRITICAL RISK**")
        st.caption(f"Fraud probability: {fraud_prob:.1f}%")

    st.markdown("---")

    # Row 4: Balance Flow Visualization
    st.markdown("### Balance Flow")
    col1, col2 = st.columns(2)

    with col1:
        # Origin Account Flow
        st.markdown("**Origin Account:**")
        st.info(f"Before: LKR {oldbalanceOrg:,.2f}")
        st.info(f"After: LKR {newBalanceOrig:,.2f}")
        orig_delta = newBalanceOrig - oldbalanceOrg
        st.metric(
            "Change",
            f"LKR {orig_delta:+,.2f}",
            delta=f"{orig_delta / oldbalanceOrg * 100:.1f}%" if oldbalanceOrg > 0 else None,
        )

    with col2:
        # Destination Account Flow
        st.markdown("**Destination Account:**")
        st.info(f"Before: LKR {oldBalanceDest:,.2f}")
        st.info(f"After: LKR {newBalanceDest:,.2f}")
        dest_delta = newBalanceDest - oldBalanceDest
        st.metric(
            "Change",
            f"LKR {dest_delta:+,.2f}",
            delta=f"{dest_delta / oldBalanceDest * 100:.1f}%" if oldBalanceDest > 0 else None,
        )

    st.markdown("---")

    # Row 5: Sankey Diagram for Money Flow
    st.markdown("### Money Flow Visualization")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Origin Account", "Destination Account"],
            color=["#FF6B6B" if prediction == 1 else "#51CF66", "#4ECDC4"]
        ),
        link=dict(
            source=[0],
            target=[1],
            value=[amount]
        )
    )])

    fig.update_layout(
        title="Transaction Flow",
        font=dict(size=12),
        height=300
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Row 6: Detailed Report
    st.markdown("### Report")

    report_data = {
        "Metric": [
            "Transaction Type",
            "Amount",
            "Origin Balance (Initial)",
            "Origin Balance (Final)",
            "Destination Balance (Initial)",
            "Destination Balance (Final)",
            "Prediction",
            "Model Confidence",
            "Fraud Probability"
        ],
        "Value": [
            transaction_type,
            f"LKR {amount:,.2f}",
            f"LKR {oldbalanceOrg:,.2f}",
            f"LKR {newBalanceOrig:,.2f}",
            f"LKR {oldBalanceDest:,.2f}",
            f"LKR {newBalanceDest:,.2f}",
            "Fraudulent" if prediction == 1 else "Legitimate",
            f"{confidence:.1f}%",
            f"{fraud_prob:.1f}%"
        ]
    }

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, width="stretch", hide_index=True)
