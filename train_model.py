import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Generator-based weights (source of truth) ──────────────────────────────
# These are the relative coefficients from generate_data.py's churn rule set,
# normalised to sum to 1. The Random Forest below is used purely to VALIDATE
# that the signal is recoverable from the noisy synthetic data — its raw
# feature_importances_ are NOT used as the headline weights because threshold-
# based rules leak importance to high-cardinality continuous features.

GENERATOR_COEFFICIENTS = {
    "lastLogin":  4.0,   # last_login_days_ago > 60
    "nps":        3.0,   # nps_score < 6
    "tickets":    2.5,   # support_tickets_last_90d > 10
    "tenure":     2.0,   # tenure_months < 12
    "onboarding": 2.0,   # onboarding_completed == 0
    "contract":   1.5,   # contract_type == "Monthly"
    "modules":    1.5,   # product_modules_used < 4
    "industry":   1.5,   # industry risk (max coefficient in map)
    "spend":      1.0,   # monthly_spend < 2000
    "csm":        1.0,   # has_dedicated_csm == 0
    "users":      1.0,   # num_users < 20
}

def normalised_weights():
    total = sum(GENERATOR_COEFFICIENTS.values())
    return {k: round(v / total, 3) for k, v in GENERATOR_COEFFICIENTS.items()}


def train_and_export():
    print("Loading data from data/clients.csv...")
    try:
        df = pd.read_csv("data/clients.csv")
    except FileNotFoundError:
        print("Error: data/clients.csv not found. Please run generate_data.py first.")
        return

    # Categorical encoding
    contract_mapping = {"Multi-year": 0, "Annual": 1, "Monthly": 2}
    industry_mapping = {"HR-Tech": 0, "Fintech": 1, "Logistics": 2, "Healthcare": 3, "Retail": 4}

    df["contract_encoded"] = df["contract_type"].map(contract_mapping)
    df["industry_encoded"] = df["industry"].map(industry_mapping)

    features = [
        "last_login_days_ago",
        "nps_score",
        "support_tickets_last_90d",
        "tenure_months",
        "onboarding_completed",
        "contract_encoded",
        "product_modules_used",
        "industry_encoded",
        "monthly_spend",
        "has_dedicated_csm",
        "num_users",
    ]

    X = df[features]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ── Evaluation (validation only) ───────────────────────────────────────
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation (Test Set) ---")
    print("WARNING: Because labels are synthetically generated from explicit rules,")
    print("these test metrics mainly validate that the Random Forest correctly learned")
    print("the pipeline's logic. This is NOT a claim of real-world predictive performance.")
    print("-" * 50)
    print(f"Accuracy:  {acc:.1%}")
    print(f"Precision: {prec:.1%}")
    print(f"Recall:    {rec:.1%}")
    print("Confusion Matrix:")
    print(f"  TN: {cm[0][0]}  |  FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]}  |  TP: {cm[1][1]}")

    # ── Export generator-derived weights (NOT raw feature_importances_) ─────
    weights = normalised_weights()

    os.makedirs("model", exist_ok=True)
    weights_path = "model/feature_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\nGenerator-derived feature weights exported to: {weights_path}")
    print("(These are normalised from the coefficients in generate_data.py,")
    print(" NOT the Random Forest's raw feature_importances_.)")
    print()
    print("Feature weights written:")
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {k:>12s}: {v:.3f}")


if __name__ == "__main__":
    train_and_export()
