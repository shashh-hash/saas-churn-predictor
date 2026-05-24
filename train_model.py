import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def train_and_export():
    print("Loading data from data/clients.csv...")
    try:
        df = pd.read_csv("data/clients.csv")
    except FileNotFoundError:
        print("Error: data/clients.csv not found. Please run generate_data.py first.")
        return

    # Categorical encoding
    # We map to numeric explicitly to ensure stable feature interpretation
    contract_mapping = {"Multi-year": 0, "Annual": 1, "Monthly": 2}
    industry_mapping = {"HR-Tech": 0, "Fintech": 1, "Logistics": 2, "Healthcare": 3, "Retail": 4}
    
    df["contract_encoded"] = df["contract_type"].map(contract_mapping)
    df["industry_encoded"] = df["industry"].map(industry_mapping)

    features = [
        "last_login_days_ago",
        "nps_score",
        "support_tickets_last_90d",
        "monthly_spend",
        "tenure_months",
        "product_modules_used",
        "num_users",
        "contract_encoded",
        "onboarding_completed",
        "has_dedicated_csm",
        "industry_encoded"
    ]

    X = df[features]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation (Test Set) ---")
    print("WARNING: Because labels are synthetically generated from explicit rules,")
    print("these test metrics mainly validate that the Random Forest correctly learned")
    print("the pipeline's logic. This is NOT a claim of real-world predictive performance.")
    print("-" * 35)
    print(f"Accuracy:  {acc:.1%}")
    print(f"Precision: {prec:.1%}")
    print(f"Recall:    {rec:.1%}")
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

    # Extract feature importances
    importances = model.feature_importances_
    
    # We rename the keys to match the frontend expectations where possible
    frontend_keys = [
        "lastLogin",
        "nps",
        "tickets",
        "spend",
        "tenure",
        "modules",
        "users",
        "contract",
        "onboarding",
        "csm",
        "industry"
    ]
    
    weights = {k: float(v) for k, v in zip(frontend_keys, importances)}

    # Save to model/feature_weights.json
    os.makedirs("model", exist_ok=True)
    weights_path = "model/feature_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
        
    print(f"\nFeature weights exported to: {weights_path}")

if __name__ == "__main__":
    train_and_export()
