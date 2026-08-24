# Payment Fraud-Risk Scorer — Track 02: AI Risk Manager

A simple, **defense-only** fraud detector for payment transactions, built on the
Kaggle *Online Payments Fraud Detection* (PaySim) dataset — 6.36M transactions,
0.13% fraud.

- **ML** — scikit-learn: a `LogisticRegression` baseline and a
  `HistGradientBoostingClassifier` scorer.
- **AI** — an LLM (Groq) turns each flagged transaction into a plain-English
  reason and a defensive action recommendation.

Everything lives in two files: **`fraud_risk_scorer.py`** (the ML + AI pipeline)
and **`app.py`** (a Streamlit web app that reuses it).

## How it works

1. **Features** — the balance-consistency errors (`errorBalanceOrig`,
   `origEmptied`, …) are the strongest fraud signals in this data.
2. **Honest split** — train on the earlier timeline, test on the *future*
   (`step`-based), so metrics reflect unseen transactions, not leaked ones.
3. **Models** — logistic baseline + gradient boosting, both class-weighted for
   the 0.13% fraud rate.
4. **Metrics** — PR-AUC and ROC-AUC on the held-out set (PR-AUC is the honest
   metric at this imbalance).
5. **Cost-based threshold** — we pick the cut-off that minimises total loss
   = *missed-fraud amount + (false alarms × `FP_COST`)*. Raising `FP_COST`
   demands higher precision — the false-positive trade-off is explicit.
6. **Explanations** — top drivers come from the logistic coefficients; the LLM
   phrases them for a merchant and recommends a defensive action.

**Defense-only:** the system only scores and recommends defensive actions
(manual review, step-up auth, hold settlement). No protected attributes, nothing
offense-capable.

## Run

```bash
pip install -r requirements.txt

# optional — enables the LLM explanations (free key: https://console.groq.com)
export GROQ_API_KEY=gsk_...        # Windows: set GROQ_API_KEY=gsk_...

python fraud_risk_scorer.py        # command-line version (prints metrics)
streamlit run app.py               # interactive web app
```

Keep `PS_20174392719_1491204439457_log.csv` in the same folder as the scripts.
Without a Groq key everything still runs and shows template explanations.

## Deploy the app (free)

Streamlit needs a persistent server, so deploy on **Streamlit Community Cloud**
(https://share.streamlit.io) — not Vercel. Push this repo to GitHub, point it at
`app.py`, and add `GROQ_API_KEY` under the app's Secrets. Hugging Face Spaces and
Render work too.
