import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="FraudX",
    page_icon="images/FraudX.svg",
    layout="wide"
)
# ===========================================================

model = joblib.load('fraud_detection_model.pkl')

# ==================== SESSION STATE INITIALIZATION ====================
if 'transaction_type' not in st.session_state:
    st.session_state.transaction_type = "PAYMENT"
if 'amount' not in st.session_state:
    st.session_state.amount = 1000.0
if 'oldbalanceOrg' not in st.session_state:
    st.session_state.oldbalanceOrg = 10000.0
if 'newBalanceOrig' not in st.session_state:
    st.session_state.newBalanceOrig = 9000.0
if 'oldBalanceDest' not in st.session_state:
    st.session_state.oldBalanceDest = 5000.0
if 'newBalanceDest' not in st.session_state:
    st.session_state.newBalanceDest = 0.0
# ===========================================================

# ==================== VALIDATION LAYER ====================
def validate_transaction(transaction_type, amount, oldbalanceOrg, newBalanceOrig, 
                        oldBalanceDest, newBalanceDest):
    """
    Validates transaction data for logical inconsistencies.
    Returns: errors (list), warnings (list)
    """
    errors = []
    warnings = []
    
    # ❌ ERRORS - STOP EXECUTION
    
    # Check for negative values
    if amount < 0 or oldbalanceOrg < 0 or newBalanceOrig < 0 or oldBalanceDest < 0 or newBalanceDest < 0:
        errors.append("Negative values are not allowed.")
    
    # Check balance consistency at origin
    if oldbalanceOrg == newBalanceOrig and amount > 0:
        errors.append("No deduction from origin account (balance unchanged).")
    
    # Check money credited without debit
    origin_change = oldbalanceOrg - newBalanceOrig
    dest_change = newBalanceDest - oldBalanceDest
    
    if origin_change <= 0 and dest_change > 0:
        errors.append("Money credited to destination without deduction from origin.")
    
    # ⚠️ WARNINGS - ALERT BUT CONTINUE
    
    # Check if balance changes match amount
    if origin_change != amount:
        warnings.append(f"Origin balance change ({origin_change}) doesn't match transaction amount ({amount}).")
    
    if dest_change != amount and dest_change > 0:
        warnings.append(f"Destination balance change ({dest_change}) doesn't match transaction amount ({amount}).")
    
    # High-risk transactions
    if transaction_type in ["TRANSFER", "CASH_OUT"] and amount > 50000:
        warnings.append(f"High-risk transaction: {transaction_type} with large amount (LKR {amount:,.2f}).")
    
    return errors, warnings
# ===========================================================

# ==================== HEADER WITH LOGO ====================
col1, col2 = st.columns([1, 4])

with col1:
    st.image("images/FraudX.svg", width=200)

with col2:
    st.title("Advanced Fraud Detection System")
    # st.markdown("Powered by Machine Learning to identify fraudulent transactions in real-time.")
    st.markdown("Powered by Ekek lept team.")
st.divider()

st.markdown("### Please input the transaction details below")

transaction_type = st.selectbox(
    "Transaction Type", 
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
    index=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"].index(st.session_state.transaction_type),
    key="transaction_type"
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=st.session_state.amount,
    key="amount"
)
oldbalanceOrg = st.number_input(
    "Old Balance of Origin Account",
    min_value=0.0,
    value=st.session_state.oldbalanceOrg,
    key="oldbalanceOrg"
)
newBalanceOrig = st.number_input(
    "New Balance of Origin Account",
    min_value=0.0,
    value=st.session_state.newBalanceOrig,
    key="newBalanceOrig"
)

oldBalanceDest = st.number_input(
    "Old Balance of Destination Account",
    min_value=0.0,
    value=st.session_state.oldBalanceDest,
    key="oldBalanceDest"
)
newBalanceDest = st.number_input(
    "New Balance of Destination Account",
    min_value=0.0,
    value=st.session_state.newBalanceDest,
    key="newBalanceDest"
)



if st.button("Predict"):
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
    
    # Proceed with prediction
    input_data = pd.DataFrame(
        [
            {
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newBalanceOrig,  # Changed from newBalanceOrig
                "oldbalanceDest": oldBalanceDest,  # Changed from oldBalanceDest
                "newbalanceDest": newBalanceDest,  # Changed from newBalanceDest
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
    st.markdown("###Transaction Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transaction Type", transaction_type)
    
    with col2:
        st.metric("Amount", f"LKR {amount:,.2f}")
    
    with col3:
        st.metric("Origin Balance Change", f"LKR {oldbalanceOrg - newBalanceOrig:,.2f}")
    
    with col4:
        st.metric("Destination Balance Change", f"LKR {newBalanceDest - oldBalanceDest:,.2f}")
    
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
            risk_color = "🟢"
            risk_text = "LOW RISK"
        elif fraud_prob < 50:
            risk_color = "🟡"
            risk_text = "MEDIUM RISK"
        elif fraud_prob < 75:
            risk_color = "🟠"
            risk_text = "HIGH RISK"
        else:
            risk_color = "🔴"
            risk_text = "CRITICAL RISK"
        
        st.warning(f"{risk_color} **{risk_text}**")
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
        st.metric("Change", f"-LKR {oldbalanceOrg - newBalanceOrig:,.2f}", delta=f"{((newBalanceOrig/oldbalanceOrg - 1)*100):.1f}%" if oldbalanceOrg > 0 else "N/A")
    
    with col2:
        # Destination Account Flow
        st.markdown("**Destination Account:**")
        st.info(f"Before: LKR {oldBalanceDest:,.2f}")
        st.info(f"After: LKR {newBalanceDest:,.2f}")
        st.metric("Change", f"+LKR {newBalanceDest - oldBalanceDest:,.2f}", delta=f"{((newBalanceDest/oldBalanceDest - 1)*100):.1f}%" if oldBalanceDest > 0 else "N/A")
    
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
    st.plotly_chart(fig, use_container_width=True)
    
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
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Reset Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Reset", use_container_width=True):
            # Clear session state
            st.session_state.transaction_type = "PAYMENT"
            st.session_state.amount = 00.00
            st.session_state.oldbalanceOrg = 00.0
            st.session_state.newBalanceOrig = 00.0
            st.session_state.oldBalanceDest = 00.0
            st.session_state.newBalanceDest = 00.0
            st.rerun()
