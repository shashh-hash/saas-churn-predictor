# B2B SaaS Churn Predictor 📊

An interactive machine learning web app that predicts client churn risk for B2B SaaS companies — built from hands-on experience managing enterprise accounts at Shipsy and ZingHR.

## 🚀 Live Demo
> Run locally using the instructions below.

## 💡 Motivation
After 2 years as the primary interface between enterprise clients and SaaS product teams, I noticed that churn was rarely a surprise — it was almost always preceded by the same signals: declining logins, rising support tickets, low NPS, and incomplete onboarding. This project turns those observations into a predictive model.

## 🛠️ Features
- **Dashboard** — Portfolio-level view of churn rates by contract type, industry, tenure, and NPS
- **Churn Predictor** — Input any client profile and get an instant churn probability score with recommended actions
- **Feature Insights** — Understand which signals drive churn most using Random Forest feature importance

## 🧠 Model
- Algorithm: Random Forest Classifier (scikit-learn)
- Dataset: Synthetically generated based on real B2B SaaS churn patterns (1,000 clients)
- Features: Tenure, spend, usage depth, support tickets, NPS, contract type, onboarding completion, CSM assignment
- Accuracy: ~80%+

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/shashh-hash/saas-churn-predictor.git
cd saas-churn-predictor
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

## 📁 Project Structure
```
saas-churn-predictor/
├── app.py              # Main Streamlit application
├── generate_data.py    # Synthetic dataset generator
├── requirements.txt    # Python dependencies
└── README.md
```

## 👤 Author
**Shashank Tripathi**
MiM Student @ ESCP Business School
[LinkedIn](https://linkedin.com/in/shashank-tripathi) · [GitHub](https://github.com/shashh-hash)
