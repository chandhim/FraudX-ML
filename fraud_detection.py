import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from icons import icon
from validation import validate_transaction

# ==================== PATHS & CONSTANTS ====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_detection_model.pkl"
LOGO_PATH = BASE_DIR / "images" / "FraudX.svg"
REPO_URL = "https://github.com/chandhim/FraudX"

# The logo SVG is a 375x375 canvas with the artwork in a small band;
# this viewBox crops the display to the artwork (the file is unchanged).
LOGO_VIEWBOX = "82 146 252 106"

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
TYPE_HELP = (
    "PAYMENT: pay a merchant · TRANSFER: send to another customer · "
    "CASH_OUT: withdraw via a merchant · DEBIT: withdraw to a bank · CASH_IN: deposit"
)

# Test-set results from analysis.ipynb (30% stratified hold-out, 1.9M transactions)
MODEL_RECALL = 2299 / 2464
MODEL_PRECISION = 2299 / (2299 + 102946)

INPUT_KEYS = ["transaction_type", "amount", "oldbalanceOrg", "newBalanceOrig",
              "oldBalanceDest", "newBalanceDest"]

DEFAULTS = {
    "transaction_type": "PAYMENT",
    "amount": 1000.0,
    "oldbalanceOrg": 10000.0,
    "newBalanceOrig": 9000.0,
    "oldBalanceDest": 0.0,
    "newBalanceDest": 0.0,
    "auto_balances": True,
    "result": None,
}
RESET_VALUES = {
    "transaction_type": "PAYMENT",
    "amount": 0.0,
    "oldbalanceOrg": 0.0,
    "newBalanceOrig": 0.0,
    "oldBalanceDest": 0.0,
    "newBalanceDest": 0.0,
    "auto_balances": True,
    "result": None,
}

FEATURE_GROUPS = {
    "Transaction type": lambda name: name.startswith("cat__type_"),
    "Amount": lambda name: name == "num__amount",
    "Sender balances": lambda name: name in ("num__oldbalanceOrg", "num__newbalanceOrig"),
    "Receiver balances": lambda name: name in ("num__oldbalanceDest", "num__newbalanceDest"),
}

ACCENT = "#e53935"
FRAUD_RED = "#d32f2f"
LEGIT_GREEN = "#2e7d32"
# ===========================================================

st.set_page_config(
    page_title="FraudX · Fraud Detection",
    page_icon=str(LOGO_PATH),
    layout="wide",
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_logo_svg(path, viewbox, height):
    svg = Path(path).read_text(encoding="utf-8")
    svg = re.sub(r'viewBox="[^"]*"', f'viewBox="{viewbox}"', svg, count=1)
    svg = re.sub(r'\swidth="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sheight="[^"]*"', f' height="{height}"', svg, count=1)
    svg = svg.replace("<svg ", '<svg class="fx-logo" ', 1)
    return svg


model = load_model(MODEL_PATH)


def heading(icon_name, text, size=""):
    """Section heading with a Lucide icon."""
    st.markdown(
        f'<div class="fx-heading {size}">{icon(icon_name)}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# ==================== CALLBACKS ====================
# Callbacks run before widgets are re-created, the only point where
# widget-backed session state may be changed.
def reset_inputs():
    for key, value in RESET_VALUES.items():
        st.session_state[key] = value


def analyze_current():
    st.session_state.result = analyze(*(st.session_state[k] for k in INPUT_KEYS))
# ===========================================================


# ==================== MODEL LOGIC ====================
def analyze(transaction_type, amount, old_orig, new_orig, old_dest, new_dest):
    inputs = (transaction_type, amount, old_orig, new_orig, old_dest, new_dest)
    errors, warnings = validate_transaction(*inputs)
    if errors:
        return {"inputs": inputs, "errors": errors}

    X = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
    }])
    fraud_prob = float(model.predict_proba(X)[0, 1])
    return {
        "inputs": inputs,
        "errors": [],
        "warnings": warnings,
        "is_fraud": bool(model.predict(X)[0] == 1),
        "fraud_prob": fraud_prob,
        "drivers": explain(X),
    }


def explain(X):
    """Per-group contribution to the logistic regression's log-odds of fraud.

    Each feature contributes coefficient x scaled value. Correlated features
    (old/new balance of the same account) are summed, because individually
    they are large and cancel each other out.
    """
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    z = prep.transform(X)
    z = np.asarray(z.todense() if hasattr(z, "todense") else z)[0]
    contributions = z * clf.coef_[0]
    names = prep.get_feature_names_out()
    return {
        group: float(sum(c for n, c in zip(names, contributions) if matches(n)))
        for group, matches in FEATURE_GROUPS.items()
    }


def risk_level(prob):
    if prob < 0.25:
        return "Low risk"
    if prob < 0.50:
        return "Medium risk"
    if prob < 0.75:
        return "High risk"
    return "Critical risk"


def auto_new_balances(transaction_type, amount, old_orig, old_dest):
    """New balances implied by the amount (what a well-formed record looks like)."""
    if transaction_type == "CASH_IN":
        return old_orig + amount, max(old_dest - amount, 0.0)
    new_orig = max(old_orig - amount, 0.0)
    new_dest = old_dest if transaction_type == "PAYMENT" else old_dest + amount
    return new_orig, new_dest
# ===========================================================


# ==================== STYLES ====================
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
      .fx-hero {{
        display: flex; align-items: center; gap: 1.75rem; flex-wrap: wrap;
        background: linear-gradient(135deg, #0b1220 0%, #1e293b 100%);
        border-radius: 16px; padding: 1.25rem 1.75rem; margin-bottom: 1.25rem;
        border: 1px solid rgba(148, 163, 184, .25);
      }}
      .fx-logo {{ flex: none; max-width: 100%; }}
      .fx-hero-title {{ color: #fff; font-size: 1.65rem; font-weight: 700; line-height: 1.2; }}
      .fx-hero-sub {{ color: #cbd5e1; margin-top: .25rem; }}
      .fx-badges {{ margin-top: .6rem; display: flex; gap: .4rem; flex-wrap: wrap; }}
      .fx-badges span {{
        background: rgba(255,255,255,.08); color: #e2e8f0; border: 1px solid rgba(255,255,255,.15);
        border-radius: 999px; padding: .15rem .65rem; font-size: .8rem;
      }}
      .fx-verdict {{ border-radius: 14px; padding: 1.1rem 1.35rem; color: #fff; margin-bottom: 1rem; }}
      .fx-verdict.fraud {{ background: {FRAUD_RED}; }}
      .fx-verdict.legit {{ background: {LEGIT_GREEN}; }}
      .fx-verdict-label {{ font-size: .85rem; text-transform: uppercase; letter-spacing: .06em; opacity: .9; }}
      .fx-verdict-main {{ font-size: 1.6rem; font-weight: 700; margin-top: .1rem; }}
      .fx-verdict-score {{ display: flex; align-items: baseline; gap: .6rem; margin-top: .4rem; }}
      .fx-verdict-score b {{ font-size: 2.4rem; font-weight: 800; line-height: 1; }}
      .fx-verdict-score span {{ opacity: .9; }}
      .fx-verdict-head {{ display: flex; align-items: center; gap: .5rem; }}
      .fx-heading {{
        display: flex; align-items: center; gap: .55rem;
        font-size: 1.25rem; font-weight: 600; margin: .4rem 0 .6rem;
      }}
      .fx-heading.sm {{ font-size: 1rem; margin: .9rem 0 .4rem; }}
      .fx-heading.xs {{ font-size: .95rem; margin: .1rem 0 .35rem; }}
      .fx-heading .fx-icon {{ color: {ACCENT}; flex: none; }}
      .fx-badges span {{ display: inline-flex; align-items: center; gap: .35rem; }}
      .fx-link {{ display: inline-flex; align-items: center; gap: .4rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)
# ===========================================================

# ==================== SIDEBAR: ABOUT THE PROJECT ====================
with st.sidebar:
    heading("info", "About FraudX")
    st.markdown(
        "An end-to-end machine learning project: exploratory analysis, a "
        "trained classifier, and this interactive app for scoring "
        "mobile-money transactions."
    )

    heading("workflow", "How it works", "sm")
    st.markdown(
        "- **Data:** PaySim, 6.36M simulated mobile-money transactions, 0.13% fraud\n"
        "- **Features:** transaction type, amount, sender and receiver balances before and after\n"
        "- **Model:** scikit-learn pipeline, scaling + one-hot encoding + "
        "logistic regression with balanced class weights"
    )

    heading("flask-conical", "Test-set results", "sm")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{MODEL_RECALL:.0%}", help="Share of real fraud the model catches.")
    c2.metric("Precision", f"{MODEL_PRECISION:.0%}", help="Share of flagged transactions that are really fraud.")
    st.caption(
        "Tuned to miss as little fraud as possible, so it raises many false alarms. "
        "A flag means *needs review*, not a final verdict."
    )

    heading("rocket", "Next steps", "sm")
    st.markdown(
        "- Gradient-boosted trees\n"
        "- Balance-error features\n"
        "- Decision threshold tuned on the precision–recall curve"
    )

    st.divider()
    st.markdown("**Built with** Python · scikit-learn · pandas · Streamlit · Plotly")
    st.markdown(
        f'<a class="fx-link" href="{REPO_URL}" target="_blank">{icon("github", 16)} Source code on GitHub</a>',
        unsafe_allow_html=True,
    )
    st.caption("Switch between light and dark mode from the ⋮ menu at the top right.")
# ===========================================================

# ==================== HEADER ====================
st.markdown(
    f"""
    <div class="fx-hero">
      {load_logo_svg(LOGO_PATH, LOGO_VIEWBOX, 64)}
      <div>
        <div class="fx-hero-title">Transaction Fraud Detection</div>
        <div class="fx-hero-sub">Score a mobile-money transaction with a machine learning model and see why it was flagged.</div>
        <div class="fx-badges"><span>{icon("database", 14)}6.36M transactions</span><span>{icon("brain-circuit", 14)}Logistic regression</span><span>{icon("target", 14)}{MODEL_RECALL:.0%} fraud recall</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ===========================================================

form_col, result_col = st.columns([1.1, 1], gap="large")

# ==================== LEFT: INPUT ====================
with form_col:
    heading("square-pen", "Enter a transaction")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        transaction_type = c1.selectbox("Transaction type", TRANSACTION_TYPES, key="transaction_type", help=TYPE_HELP)
        amount = c2.number_input("Amount", min_value=0.0, step=100.0, format="%.2f", key="amount")

        auto = st.toggle(
            "Calculate new balances from the amount", key="auto_balances",
            help="Turn off to enter balances by hand, e.g. to test records that don't add up.",
        )

        heading("wallet", "Sender account", "xs")
        c1, c2 = st.columns(2)
        old_orig = c1.number_input("Balance before", min_value=0.0, step=100.0, format="%.2f", key="oldbalanceOrg")
        heading("user", "Receiver account", "xs")
        d1, d2 = st.columns(2)
        old_dest = d1.number_input("Balance before", min_value=0.0, step=100.0, format="%.2f", key="oldBalanceDest")

        if auto:
            # Allowed: these widgets have not been created yet in this run.
            st.session_state.newBalanceOrig, st.session_state.newBalanceDest = auto_new_balances(
                transaction_type, amount, old_orig, old_dest
            )
        c2.number_input("Balance after", min_value=0.0, step=100.0, format="%.2f", key="newBalanceOrig", disabled=auto)
        d2.number_input("Balance after", min_value=0.0, step=100.0, format="%.2f", key="newBalanceDest", disabled=auto)

    b1, b2 = st.columns([3, 1])
    b1.button("Analyze transaction", icon=":material/search:", type="primary", on_click=analyze_current, width="stretch")
    b2.button("Reset", icon=":material/restart_alt:", on_click=reset_inputs, width="stretch")
# ===========================================================

# ==================== RIGHT: RESULT ====================
with result_col:
    heading("activity", "Result")
    current_inputs = tuple(st.session_state[k] for k in INPUT_KEYS)
    result = st.session_state.result

    if result is None:
        st.info("Enter a transaction and click **Analyze transaction**.", icon=":material/touch_app:")
    elif result["inputs"] != current_inputs:
        st.info("The inputs have changed. Click **Analyze transaction** to update the result.", icon=":material/refresh:")
    elif result["errors"]:
        for error in result["errors"]:
            st.error(error, icon=":material/error:")
    else:
        prob = result["fraud_prob"]
        kind = "fraud" if result["is_fraud"] else "legit"
        verdict = "Likely fraud" if result["is_fraud"] else "Looks legitimate"
        verdict_icon = "shield-alert" if result["is_fraud"] else "shield-check"
        st.markdown(
            f"""
            <div class="fx-verdict {kind}">
              <div class="fx-verdict-head">{icon(verdict_icon, 20)}<span class="fx-verdict-label">Model verdict</span></div>
              <div class="fx-verdict-main">{verdict}</div>
              <div class="fx-verdict-score"><b>{prob:.0%}</b><span>fraud score · {risk_level(prob)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        heading("chart-column", "What drove this score", "xs")
        drivers = sorted(result["drivers"].items(), key=lambda kv: abs(kv[1]))
        fig = go.Figure(go.Bar(
            x=[v for _, v in drivers],
            y=[k for k, _ in drivers],
            orientation="h",
            marker_color=[FRAUD_RED if v > 0 else LEGIT_GREEN for _, v in drivers],
            hovertemplate="%{y}: %{x:+.1f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_width=1, line_color="#888")
        fig.update_layout(
            height=210, margin=dict(l=0, r=10, t=5, b=40),
            xaxis_title="← towards legitimate · towards fraud →",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Each bar is that input's contribution to the model's log-odds of fraud.")

        if result["warnings"]:
            with st.container(border=True):
                heading("list-checks", "Data checks", "xs")
                for warning in result["warnings"]:
                    st.warning(warning, icon=":material/warning:")

        tx_type, amt, o_old, o_new, d_old, d_new = result["inputs"]
        with st.expander("Transaction details"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Before": [o_old, d_old],
                        "After": [o_new, d_new],
                        "Change": [o_new - o_old, d_new - d_old],
                    },
                    index=["Sender", "Receiver"],
                ).style.format("{:+,.2f}", subset=["Change"]).format("{:,.2f}", subset=["Before", "After"]),
                width="stretch",
            )
            st.caption(f"{tx_type} of {amt:,.2f}. Amounts are simulated units from the PaySim dataset.")

        st.caption(
            f"Model note: catches {MODEL_RECALL:.0%} of fraud on the test set, but only "
            f"{MODEL_PRECISION:.0%} of its flags are real fraud. See the sidebar for details."
        )
# ===========================================================
