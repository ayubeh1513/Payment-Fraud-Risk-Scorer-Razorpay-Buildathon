import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

import fraud_risk_scorer as frs   
st.set_page_config(page_title="Payment Fraud-Risk Scorer", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading data and training the model…")
def build():
    df = frs.load_and_prepare()
    train_df, test_df = frs.time_split(df)
    logit, scaler, model, base_proba, proba, yte = frs.train(train_df, test_df)
    test_df = test_df.assign(score=proba)
    return test_df, logit, scaler, base_proba, proba, yte


test_df, logit, scaler, base_proba, proba, yte = build()
amounts = test_df.amount.values

st.title("🛡️ Payment Fraud-Risk Scorer")
st.caption("Track 02 · AI Risk Manager — **defense-only**: scores transactions and "
           "recommends defensive actions. No offense-capable output.")

c1, c2, c3 = st.columns(3)
c1.metric("PR-AUC (held-out)", f"{average_precision_score(yte, proba):.3f}")
c2.metric("ROC-AUC (held-out)", f"{roc_auc_score(yte, proba):.3f}")
c3.metric("Fraud rate", f"{yte.mean():.3%}")

st.sidebar.header("Cost model")
fp_cost = st.sidebar.slider("Cost per false alarm (₹)", 50, 20000, 500, 50,
                            help="Friction/ops cost of reviewing a legit payment. "
                                 "Higher = the model demands higher precision.")

thresholds = np.linspace(0.05, 0.95, 19)
rows = []
for t in thresholds:
    pred = proba >= t
    tp = int((pred & (yte == 1)).sum()); fp = int((pred & (yte == 0)).sum())
    fn = int((~pred & (yte == 1)).sum())
    loss = amounts[(~pred) & (yte == 1)].sum() + fp_cost * fp
    rows.append({"threshold": round(t, 2), "precision": tp / max(tp + fp, 1),
                 "recall": tp / max(tp + fn, 1), "false_alarms": fp,
                 "missed_fraud": fn, "loss": loss})
curve = pd.DataFrame(rows)
best = curve.loc[curve["loss"].idxmin()]
thr = float(best["threshold"])
no_model = amounts[yte == 1].sum()

st.sidebar.markdown(
    f"**Cost-optimal threshold {thr:.2f}**\n\n"
    f"- precision **{best['precision']:.3f}**\n- recall **{best['recall']:.3f}**\n"
    f"- false alarms **{int(best['false_alarms']):,}**\n"
    f"- missed fraud **{int(best['missed_fraud']):,}**\n"
    f"- loss avoided **{(no_model - best['loss']) / no_model:.1%}**")

left, right = st.columns(2)
with left:
    st.subheader("Precision & recall vs threshold")
    st.line_chart(curve.set_index("threshold")[["precision", "recall"]])
with right:
    st.subheader("Expected loss vs threshold (₹)")
    st.line_chart(curve.set_index("threshold")[["loss"]])

st.subheader("🚩 Flagged transactions")
flagged = test_df[test_df.score >= thr].sort_values("score", ascending=False).copy()
flagged["risk"] = flagged["score"].round(3)
flagged["action"] = flagged["score"].apply(frs.defensive_action)
flagged["fraud?"] = flagged["isFraud"].map({1: "✅ fraud", 0: "legit"})
st.caption(f"{len(flagged):,} transactions flagged at threshold {thr:.2f}.")
st.dataframe(flagged[["type", "amount", "risk", "fraud?", "action"]].head(200),
             width="stretch", height=300)

st.subheader("🔎 Score a transaction")
with st.form("score"):
    a, b, c = st.columns(3)
    ttype = a.selectbox("type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
    amount = b.number_input("amount (₹)", 0.0, value=181000.0, step=1000.0)
    old_org = c.number_input("sender balance before", 0.0, value=181000.0, step=1000.0)
    d, e, f = st.columns(3)
    new_org = d.number_input("sender balance after", 0.0, value=0.0, step=1000.0)
    old_dest = e.number_input("receiver balance before", 0.0, value=0.0, step=1000.0)
    new_dest = f.number_input("receiver balance after", 0.0, value=0.0, step=1000.0)
    go = st.form_submit_button("Score")

if go:
    row = pd.Series({
        "type": ttype, "amount": amount,
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "errorBalanceOrig": new_org + amount - old_org,
        "errorBalanceDest": old_dest + amount - new_dest,
        "origEmptied": int(old_org > 0 and new_org == 0),
        "destWasEmpty": int(old_dest == 0),
        "amtToOrigBal": amount / (old_org + 1.0),
        "isTransfer": int(ttype == "TRANSFER"), "isCashout": int(ttype == "CASH_OUT")})
    score = float(logit.predict_proba(scaler.transform(row[frs.FEATURES].values.reshape(1, -1)))[0, 1])
    action = frs.defensive_action(score)
    k1, k2 = st.columns([1, 3])
    k1.metric("Risk score", f"{score:.2f}")
    (k2.error if score >= thr else k2.success)(
        f"{'FLAGGED — ' + action if score >= thr else 'Allow'} (threshold {thr:.2f})")
    txn = {"type": ttype, "amount": amount, "score": score}
    st.info(frs.ai_explanation(txn, frs.top_drivers(logit, scaler, row), action))

st.divider()
