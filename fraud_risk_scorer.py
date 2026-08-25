import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import average_precision_score, roc_auc_score

_HERE = Path(__file__).parent
CSV_FILE = next((_HERE / n for n in ("PS_20174392719_1491204439457_log.csv",
                                      "sample_transactions.csv")
                 if (_HERE / n).exists()), _HERE / "sample_transactions.csv")
FP_COST  = 500.0       
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-20b"

FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
            "newbalanceDest", "errorBalanceOrig", "errorBalanceDest",
            "origEmptied", "destWasEmpty", "amtToOrigBal", "isTransfer", "isCashout"]


def load_and_prepare():
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df):,} transactions | fraud rate {df.isFraud.mean():.4%}")

    df["errorBalanceOrig"] = df.newbalanceOrig + df.amount - df.oldbalanceOrg
    df["errorBalanceDest"] = df.oldbalanceDest + df.amount - df.newbalanceDest
    df["origEmptied"] = ((df.oldbalanceOrg > 0) & (df.newbalanceOrig == 0)).astype(int)
    df["destWasEmpty"] = (df.oldbalanceDest == 0).astype(int)
    df["amtToOrigBal"] = df.amount / (df.oldbalanceOrg + 1.0)
    df["isTransfer"] = (df.type == "TRANSFER").astype(int)
    df["isCashout"] = (df.type == "CASH_OUT").astype(int)
    return df


def time_split(df):
    cut = df.step.quantile(0.70)
    train, test = df[df.step <= cut], df[df.step > cut]
    print(f"Train {len(train):,} | Test (future) {len(test):,}")
    return train, test


def train(train, test):
    Xtr, ytr = train[FEATURES].values, train.isFraud.values
    Xte, yte = test[FEATURES].values, test.isFraud.values

    scaler = StandardScaler().fit(Xtr)
    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(scaler.transform(Xtr), ytr)

    weights = compute_sample_weight("balanced", ytr)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=48, random_state=42)
    model.fit(Xtr, ytr, sample_weight=weights)

    base_proba = logit.predict_proba(scaler.transform(Xte))[:, 1]
    proba = model.predict_proba(Xte)[:, 1]
    return logit, scaler, model, base_proba, proba, yte


def evaluate(base_proba, proba, yte, amounts):
    print("\n--- Held-out metrics ---")
    print(f"Logistic baseline : PR-AUC {average_precision_score(yte, base_proba):.3f} | "
          f"ROC-AUC {roc_auc_score(yte, base_proba):.3f}")
    print(f"Gradient boosting : PR-AUC {average_precision_score(yte, proba):.3f} | "
          f"ROC-AUC {roc_auc_score(yte, proba):.3f}")

    best_thr, best_loss = 0.5, float("inf")
    for thr in np.linspace(0.05, 0.95, 19):
        pred = proba >= thr
        loss = amounts[(~pred) & (yte == 1)].sum() + FP_COST * ((pred) & (yte == 0)).sum()
        if loss < best_loss:
            best_thr, best_loss = thr, loss

    pred = proba >= best_thr
    tp = int((pred & (yte == 1)).sum()); fp = int((pred & (yte == 0)).sum())
    fn = int((~pred & (yte == 1)).sum())
    no_model_loss = amounts[yte == 1].sum()
    print(f"\nChosen threshold {best_thr:.2f} (FP cost ₹{FP_COST:,.0f})")
    print(f"Precision {tp/max(tp+fp,1):.3f} | Recall {tp/max(tp+fn,1):.3f} | "
          f"False alarms {fp:,} | Missed fraud {fn:,}")
    print(f"Loss with model ₹{best_loss:,.0f} vs ₹{no_model_loss:,.0f} without "
          f"→ avoided {(no_model_loss-best_loss)/no_model_loss:.1%}")
    return best_thr


def top_drivers(logit, scaler, row):
    x = scaler.transform(row[FEATURES].values.reshape(1, -1))[0]
    pairs = sorted(zip(FEATURES, logit.coef_[0] * x), key=lambda p: -abs(p[1]))
    return [f for f, c in pairs[:3] if c > 0]


def defensive_action(score):
    if score >= 0.90: return "HOLD settlement + manual review"
    if score >= 0.70: return "Step-up auth (OTP/2FA) + 24h hold"
    return "Async manual review; allow but flag"


def ai_explanation(txn, drivers, action):
    facts = ", ".join(drivers) or "model risk pattern"
    template = f"Flagged (risk {txn['score']:.2f}). Drivers: {facts}. Action: {action}."
    if not GROQ_API_KEY:
        return template
    prompt = (f"You are a payments fraud analyst. In 2 sentences, explain why this "
              f"transaction was flagged and justify the action. Factual, no identity "
              f"guessing.\nType {txn['type']}, amount ₹{txn['amount']:,.0f}, risk "
              f"{txn['score']:.2f}. Drivers: {facts}. Action: {action}.")
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        reply = client.chat.completions.create(model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=140)
        return reply.choices[0].message.content.strip()
    except Exception as e:     
        return f"{template}  (LLM unavailable: {type(e).__name__})"


if __name__ == "__main__":
    df = load_and_prepare()
    train_df, test_df = time_split(df)
    logit, scaler, model, base_proba, proba, yte = train(train_df, test_df)
    thr = evaluate(base_proba, proba, yte, test_df.amount.values)

    test_df = test_df.assign(score=proba)
    flagged = test_df[test_df.score >= thr].sort_values("score", ascending=False)

    print(f"\n--- Top flagged transactions ({len(flagged):,} total) ---")
    for _, row in flagged.head(5).iterrows():
        txn = {"type": row.type, "amount": row.amount, "score": row.score}
        action = defensive_action(row.score)
        print(f"\n• {row.type} ₹{row.amount:,.0f} | risk {row.score:.2f} | fraud={int(row.isFraud)}")
        print(f"  {ai_explanation(txn, top_drivers(logit, scaler, row), action)}")
