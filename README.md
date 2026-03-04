# B2B SaaS Churn Predictor 📊

An interactive machine learning web app that predicts client churn risk for B2B SaaS companies — built from hands-on experience managing enterprise accounts at Shipsy and ZingHR.

## 🚀 Live Demo
👉 **[Open the app](https://saas-churn-predictor.streamlit.app)**

## 💡 Motivation
After 2 years as the primary interface between enterprise clients and SaaS product teams at Shipsy (logistics SaaS) and ZingHR (HR-tech), I noticed that churn was rarely a surprise. It was almost always preceded by the same signals: declining logins, rising support tickets, low NPS scores, and incomplete onboarding.

This project turns those real-world observations into a predictive model. The dataset is synthetically generated, but the churn logic is grounded in actual patterns observed across 15+ enterprise accounts — factors like contract type, CSM assignment, and product adoption depth are weighted based on what genuinely moved the needle in practice.

## 🛠️ Features
- **Dashboard** — Portfolio-level view of churn rates by contract type, industry, tenure, and NPS
- **Churn Predictor** — Input any client profile and get an instant churn probability score with recommended actions
- **Feature Insights** — Understand which signals drive churn most using Random Forest feature importance

## 🧠 Model
- Algorithm: Random Forest Classifier (scikit-learn)
- Dataset: Synthetically generated (1,000 clients) with churn logic grounded in real B2B SaaS account management experience
- Features: Tenure, monthly spend, number of users, support tickets, product modules used, last login, NPS score, contract type, onboarding completion, CSM assignment, industry
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
[LinkedIn](https://www.linkedin.com/in/shashank-tripathi-a46679190/) · [GitHub](https://github.com/shashh-hash)
