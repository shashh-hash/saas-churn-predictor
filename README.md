# B2B SaaS Churn Predictor 📊

An interactive machine learning web app that predicts client churn risk for B2B SaaS companies, running entirely in the browser using pre-trained feature weights.

## 🚀 Live Demo
👉 **[Open the app](https://shashh-hash.github.io/saas-churn-predictor/)**

## 💡 Motivation
After 2 years as the primary interface between enterprise clients and SaaS product teams at Shipsy (logistics SaaS) and ZingHR (HR-tech), I noticed that churn was rarely a surprise. It was almost always preceded by the same signals: declining logins, rising support tickets, low NPS scores, and incomplete onboarding.

This project turns those real-world observations into a predictive model. The dataset is synthetically generated, but the churn logic is grounded in actual patterns observed across 15+ real enterprise accounts. Factors like contract type, CSM assignment, and product adoption depth are weighted based on what genuinely moved the needle in practice.

## 🏗 Architecture
This project uses an honest "train then deploy" pipeline:
1. **Python Data Generation**: Generates a synthetic dataset of 1,000 clients using a documented, probabilistic rule set with realistic noise (`generate_data.py`). The coefficients in this script are the single source of truth for which signals drive churn.
2. **Python Model Training**: Trains a scikit-learn Random Forest purely as a **validation step** (can the signal be recovered from noisy data?). The scoring weights exported to `model/feature_weights.json` are derived from the generator's coefficients—normalised to sum to 1—**not** from the Random Forest's raw `feature_importances_`, which are misleadingly flat due to threshold-based rules leaking importance to high-cardinality continuous columns.
3. **Static Web App**: The live GitHub Pages demo (`app.js`) is purely static JS. It fetches `model/feature_weights.json` at runtime to power the prediction engine—zero server or installation required.

## 🔑 Key Findings
Based on the generator's domain-logic coefficients (which drive the scoring weights), the signals that dominate churn prediction are:
1. **Days Since Last Login** (19.0%) — The loudest signal. Clients inactive 60+ days are the highest-risk group.
2. **NPS Score** (14.3%) — A leading early-warning sign; NPS below 6 reliably precedes churn.
3. **Support Tickets** (11.9%) — Rising tickets indicate unresolved friction.
4. **Tenure & Onboarding** (9.5% each) — Short tenure and incomplete onboarding are tied as the next strongest drivers.
5. **Contract Type, Modules Used & Industry** (7.1% each) — Monthly contracts, low module adoption, and higher-risk verticals (Retail, Healthcare) add meaningful signal.

## ⚠️ Limitations
- **Synthetic Data**: The data (`data/clients.csv`) is synthetically generated using a probabilistic rule set. 
- **Pipeline Validation, Not Real-World Claims**: The reported model accuracy metrics validate that the Random Forest correctly learned the rules used to generate the data. This accuracy is **not** a claim of real-world predictive performance.
- **Domain Logic**: The learned feature weights reflect the domain logic encoded in the data generation script rather than being learned from raw historical outcomes.

## ⚙️ Setup & Reproducibility

### 1. Clone the repo
```bash
git clone https://github.com/shashh-hash/saas-churn-predictor.git
cd saas-churn-predictor
```

### 2. Set up the Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Pipeline
Generate the data and train the model to output the feature weights:
```bash
python3 generate_data.py
python3 train_model.py
```

### 4. Open the Web App
Simply open `index.html` in your browser. It will dynamically load the weights from `model/feature_weights.json` to score churn locally.
```bash
open index.html
```

## 📁 Project Structure
```text
saas-churn-predictor/
├── data/
│   └── clients.csv             # Generated synthetic dataset
├── model/
│   └── feature_weights.json    # Generator-derived scoring weights (normalised)
├── generate_data.py            # Synthetic dataset generator
├── train_model.py              # Random Forest training script
├── index.html                  # Static web app UI
├── app.js                      # Static web app logic
├── style.css                   # Static web app styling
├── requirements.txt            # Minimal Python dependencies
└── README.md
```

## 👤 Author
**Shashank Tripathi**
MiM Student @ ESCP Business School
[LinkedIn](https://www.linkedin.com/in/shashank-tripathi-a46679190/) · [GitHub](https://github.com/shashh-hash)
