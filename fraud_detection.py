import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_model.pkl')

st.title ("Fraud Detection Application")

st.markdown("Please input the transaction details below")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH OUT", "DEBIT", "CASH IN"])

amount = st.number_input("Transaction Amount",min_value=0.0, value=1000.0,)
oldbalanceOrg = st.number_input("Old Balance of Origin Acount", min_value=0.0, value=10000.0)
newBalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, value=9000.0)

oldBalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, value=5000.0)
newBalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, value = 0.0)



if st.button("Predict"):
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

    st.subheader(f"Prediction : '{int(prediction)}'")

    if prediction == 1:
        st.error("The Transaction Can be Fraudulent.")
    else: 
        st.success("The Transaction is Legitimate.")
