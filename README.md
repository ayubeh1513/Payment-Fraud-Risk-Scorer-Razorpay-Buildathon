# Payment Fraud-Risk Scorer
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_LLM-000000?style=for-the-badge&logo=groq&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

An end-to-end **AI & ML system** built with Python and Streamlit to **score payment transactions for fraud risk, pick a cost-optimal decision threshold, and explain every flag in plain English** — turning raw transaction logs into defensible, review-ready decisions. Built for **Track 02: AI Risk Manager** — stop merchants losing money to fraud, returns and chargebacks. **Strictly defense-only.**

---

### Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Workflow](#workflow)
4. [Dataset Description](#dataset-description)
5. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
6. [Model Selection & Performance](#model-selection--performance)
7. [Cost-Based Threshold](#cost-based-threshold)
8. [Results & Business Impact](#results--business-impact)
9. [Defense-Only Design](#defense-only-design)
10. [Technology Stack](#technology-stack)
11. [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Running the App](#running-the-app)
12. [Project Structure](#project-structure)
13. [Future Scope](#future-scope)
14. [Author](#author)
15. [License](#license)

---

## Introduction
The **Payment Fraud-Risk Scorer** replaces slow, error-prone manual payment review with a transparent, intelligent, and automated alternative. Fraud is rare but expensive: in the dataset used here, only **0.13%** of transactions are fraudulent, yet each missed fraud can cost a merchant the full transaction amount, while every false alarm burns analyst time and annoys a legitimate customer.

This system tackles the problem end-to-end: from the raw **PaySim** transaction log and engineered balance-consistency features, through an honest time-based train/test split and model evaluation, to a deployed **Streamlit web application** that any risk analyst can operate — with no ML expertise required.

The project is built around **three core capabilities**:
- 🎯 **Fraud Risk Scoring** — A gradient-boosting model that assigns every transaction a calibrated fraud-risk score.
- 💰 **Cost-Optimal Thresholding** — The decision cut-off is chosen to minimise *total business loss*, with the false-positive cost as an explicit, tunable lever.
- 🤖 **AI Explanations** — A Groq-hosted LLM turns each flagged transaction into a plain-English reason and a *defensive* action recommendation.

## Key Features
* ✅ **Real-Time Fraud Scoring** — Assigns a 0–1 fraud-risk score to any transaction using a trained `HistGradientBoostingClassifier`.
* ✅ **Cost-Based Decisioning** — Automatically selects the threshold that minimises `missed-fraud amount + (false alarms × FP cost)`; raise the FP cost and the model demands higher precision.
* ✅ **Honest, Leak-Free Evaluation** — Time-based (past → future) split so reported metrics reflect genuinely unseen transactions.
* ✅ **Per-Transaction Explanations** — Top risk drivers from logistic coefficients × standardized values, phrased for a human by a Groq LLM.
* ✅ **Defensive Action Layer** — Every flag maps to a graduated defensive action (async review → step-up auth → hold settlement). Never punitive, never offense-capable.
* ✅ **Interactive Streamlit App** — Headline metrics, an FP-cost slider, live precision/recall & loss curves, a flagged-transaction queue, and a single-transaction scorer in one clean UI.
* ✅ **Runs With or Without an API Key** — LLM explanations activate when `GROQ_API_KEY` is set; otherwise a clean template fallback keeps everything working.
* ✅ **Two-File Codebase** — All ML + AI logic in `fraud_risk_scorer.py`; the Streamlit UI in `app.py` simply imports and reuses it.

## Workflow
The system follows a clear, automated five-step pipeline:

1. **Data Input** — Load the PaySim transaction log (`PS_..._log.csv`, or the bundled `sample_transactions.csv`).
2. **Feature Engineering** — Derive balance-consistency errors and behavioural flags — the strongest fraud signals in this data.
3. **Model Training** — A class-weighted `LogisticRegression` baseline/explainer and a `HistGradientBoostingClassifier` scorer, both weighted for the 0.13% fraud rate.
4. **Evaluation & Thresholding** — PR-AUC / ROC-AUC on a held-out *future* slice, then a cost-optimal threshold sweep.
5. **Score, Flag & Explain** — Flagged transactions get a defensive action and a plain-English LLM explanation, shown in the Streamlit dashboard.

```
PaySim CSV  (6.36M transactions, 0.13% fraud)
      │
      ▼
Feature Engineering  ──►  12 Model Features
  ├─ errorBalanceOrig / errorBalanceDest  (balance-consistency errors)
  ├─ origEmptied / destWasEmpty           (account-drain flags)
  ├─ amtToOrigBal                         (amount ÷ sender balance)
  └─ isTransfer / isCashout               (risky transaction types)
      │
      ▼
Honest Time-Based Split  (step ≤ 70th pct → train | > → future test)
      │
      ▼
Model Training
  ├─ LogisticRegression  (class-weighted)  → baseline + explainer
  └─ HistGradientBoosting (sample-weighted) → main risk scorer
      │
      ▼
Cost-Optimal Threshold   (minimise missed-fraud ₹ + FP_COST × false alarms)
      │
      ▼
Streamlit Web App (app.py)
  ├─ Metrics + FP-cost slider
  ├─ Precision/Recall & Loss curves
  ├─ Flagged-transaction queue + defensive action
  └─ Single-transaction scorer + Groq LLM explanation
```

## Dataset Description
**Source:** Kaggle — *Online Payments Fraud Detection* (PaySim simulator).
**Scale:** 6,362,620 transactions · fraud rate **0.129%** (highly imbalanced).

| Column | Description |
|---|---|
| `step` | Time unit (1 hour); used to build the leak-free past → future split |
| `type` | Transaction type — `TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT` |
| `amount` | Transaction amount |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before / after |
| `oldbalanceDest` / `newbalanceDest` | Receiver balance before / after |
| `isFraud` | Target label — 1 if the transaction is fraudulent |

> ⚠️ **Note on the dataset file:** the full log is ~471 MB and is **not** committed to GitHub (it exceeds GitHub's 100 MB limit and is `.gitignore`d). A **`sample_transactions.csv`** (~15 MB: every fraud case + a legit sample) is bundled so the deployed app runs. The code uses the full dataset automatically if it is present next to the scripts, else falls back to the sample. Download the full file from Kaggle for full-scale metrics.

## Data Preprocessing & Feature Engineering
### Engineered Features
| Feature | Description |
|---|---|
| `errorBalanceOrig` | `newbalanceOrig + amount − oldbalanceOrg` — sender-side balance inconsistency |
| `errorBalanceDest` | `oldbalanceDest + amount − newbalanceDest` — receiver-side balance inconsistency |
| `origEmptied` | 1 if the sender account was fully drained (`old > 0`, `new == 0`) |
| `destWasEmpty` | 1 if the receiver account started empty |
| `amtToOrigBal` | Amount as a fraction of the sender's balance |
| `isTransfer` / `isCashout` | Flags for the two transaction types where fraud concentrates |

### Pipeline Steps
| Step | Detail |
|---|---|
| 🗄️ Data Loading | `pandas.read_csv()` reads the PaySim log (path auto-resolved next to the script) |
| 🧮 Feature Engineering | Balance-consistency errors + drain/type flags computed in vectorised pandas |
| ✂️ Train/Test Split | **Time-based**: `step ≤ 70th percentile` → train, later steps → future test (no leakage) |
| ⚖️ Feature Scaling | `StandardScaler` fit on train, applied to test — used by the logistic explainer |
| 🪶 Class Imbalance | `class_weight="balanced"` (logistic) and `compute_sample_weight("balanced")` (boosting) |

### Key Findings
- **Balance-consistency errors are the dominant fraud signal** — legitimate transactions keep the balance equation consistent; fraudulent transfers/cash-outs do not.
- Fraud concentrates almost entirely in **`TRANSFER`** and **`CASH_OUT`** types.
- Fraudulent transactions frequently **empty the sender account** (`origEmptied = 1`) and target previously **empty destination accounts**.

## Model Selection & Performance
Two models are trained: a class-weighted **Logistic Regression** (fast, interpretable — used as both a baseline and the per-transaction explainer) and a **Histogram Gradient Boosting Classifier** (the main scorer). Metrics below are on the **held-out future test set** from the full 6.36M-row dataset. At a 0.13% fraud rate, **PR-AUC is the honest headline metric** — ROC-AUC looks high for almost any model on extreme imbalance.

| Model | Role | PR-AUC | ROC-AUC | Selected |
|---|---|---|---|---|
| Logistic Regression (balanced) | Baseline + explainer | ~0.70 | ~0.98 | — |
| **HistGradientBoosting (weighted)** | **Main risk scorer** | **0.746** | **0.990** | ✅ Best |

> Gradient boosting captures the non-linear balance-inconsistency patterns the linear baseline can only approximate, lifting PR-AUC — the metric that actually matters at 0.13% fraud.

## Cost-Based Threshold
Rather than defaulting to 0.5, the decision threshold is chosen to **minimise total business loss**:

```
total loss  =  (₹ amount of missed fraud)  +  (FP_COST × number of false alarms)
```

`FP_COST` — the operational cost of reviewing one legitimate payment — is the **business lever**. Raising it forces the model toward higher precision. Measured on the held-out set:

| FP cost per false alarm | Chosen threshold | Recall | Precision | Behaviour |
|---|---|---|---|---|
| ₹500 | 0.63 | **0.95** | 0.04 | Catch almost all fraud; tolerate more review load |
| ₹5,000 | 0.79 | 0.80 | **0.24** | Fewer false alarms; accept some missed fraud |

The Streamlit app exposes this as a live slider so a risk manager can dial the trade-off to their own operations.

## Results & Business Impact
| Metric | Value |
|---|---|
| 🎯 ROC-AUC (held-out) | **0.990** |
| 📈 PR-AUC (held-out) | **0.746** |
| 🔍 Recall at low FP cost | **up to 0.95** |
| ⚙️ Decision policy | **Cost-optimal, tunable** |

- The false-positive cost lever makes the **precision/recall trade-off explicit and auditable** — no hidden 0.5 cut-off.
- Every flag ships with a **plain-English reason and a defensive action**, so analysts act on decisions rather than opaque scores.
- Early, explainable fraud detection delivers **direct, measurable ROI** by stopping loss before settlement — without punishing legitimate customers.

## Defense-Only Design
This system is **strictly defense-only**, as required by the track:
- It **only scores and recommends defensive actions** — async manual review, step-up authentication (OTP/2FA), and holding settlement. Nothing punitive or offense-capable.
- It uses **no protected attributes** and does no identity guessing — features are purely transactional (amounts and balance mechanics).
- The LLM is instructed to stay **factual and avoid identity inference**, explaining *why a pattern is risky*, not *who* to punish.

## Technology Stack
* **Application:** Streamlit
* **Language:** Python 3.9+
* **ML Framework:** Scikit-learn
    * `HistGradientBoostingClassifier`, `LogisticRegression`
    * `StandardScaler`, `compute_sample_weight`, `average_precision_score`, `roc_auc_score`
* **AI Layer:** Groq LLM API — `llama-3.3-70b-versatile` (free tier)
* **Data:** Pandas + NumPy on the PaySim CSV

---

## Getting Started
Follow these instructions to set up and run the project locally.

### Prerequisites
* Python 3.9 or higher
* `pip` package manager
* (Optional) A free **Groq API key** from [console.groq.com](https://console.groq.com) to enable LLM explanations
* (Optional) The full PaySim `PS_..._log.csv` from Kaggle for full-scale metrics — otherwise the bundled `sample_transactions.csv` is used

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayubeh1513/payment-fraud-risk-scorer-razorpay-buildathon
   cd payment-fraud-risk-scorer-razorpay-buildathon
   ```
2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On macOS / Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Core packages installed:
   | Package | Purpose |
   |---|---|
   | streamlit | Web application |
   | scikit-learn | ML models & metrics |
   | pandas | Data loading & feature engineering |
   | numpy | Numerical operations |
   | groq | LLM explanation client |

4. **(Optional) Enable LLM explanations** — set your Groq API key:
   ```bash
   # macOS / Linux
   export GROQ_API_KEY=gsk_...
   # Windows (PowerShell)
   $env:GROQ_API_KEY="gsk_..."
   ```
   > 🔒 Never commit your key. On Streamlit Community Cloud, add `GROQ_API_KEY` under the app's **Secrets** instead.

### Running the App
```bash
streamlit run app.py
```
The application opens in your browser at `http://localhost:8501`.

> ⏱️ On the full dataset the first load trains on 6.36M rows (~1 minute); it is then cached, so the sliders stay instant.

**Command-line version (prints metrics + top flagged transactions):**
```bash
python fraud_risk_scorer.py
```

**Deploy for free:** push this repo to GitHub and deploy on **Streamlit Community Cloud** (https://share.streamlit.io) pointing at `app.py`, then add `GROQ_API_KEY` under Secrets. Streamlit needs a persistent server, so this is preferred over serverless hosts like Vercel.

---

## Project Structure
```
payment-fraud-risk-scorer/
│
├── fraud_risk_scorer.py       # ML + AI pipeline (features, split, train, evaluate, explain)
├── app.py                     # Streamlit web app — imports & reuses the pipeline
├── requirements.txt           # Python dependencies
├── sample_transactions.csv    # ~15MB sample so the deployed app runs (full CSV is gitignored)
├── .gitignore                 # Excludes the 471MB full dataset, __pycache__, venv
└── README.md
```

> **Note:** the full 471 MB `PS_..._log.csv`, `__pycache__/`, and virtual environments are excluded via `.gitignore`. The full dataset is downloaded separately from Kaggle for local full-scale runs.

---

## Future Scope
| Roadmap Item | Description |
|---|---|
| 📊 Probability Calibration | Add isotonic/Platt calibration so scores read as true fraud probabilities |
| 🧠 Anomaly Detection | Unsupervised models (e.g. Isolation Forest) to catch novel, previously unseen fraud patterns |
| ⏱️ Real-Time Scoring | Serve the model behind a low-latency API for inline transaction scoring |
| 🔗 Feedback Loop | Feed confirmed-fraud analyst decisions back into retraining |
| 🔍 Richer Features | Velocity/aggregation features (per-account transaction rates over time) |

---

## Author
**Ayushman Behera**
B.Tech CSE (DSML)
Lovely Professional University, Jalandhar, Punjab
🔗 GitHub: [ayubeh1513](https://github.com/ayubeh1513)

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
