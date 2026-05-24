import os
import pandas as pd
import numpy as np

np.random.seed(42)

def generate_churn_dataset(n=1000):
    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "tenure_months": np.random.randint(1, 60, n),
        "monthly_spend": np.round(np.random.uniform(500, 10000, n), 2),
        "num_users": np.random.randint(5, 200, n),
        "support_tickets_last_90d": np.random.randint(0, 30, n),
        "product_modules_used": np.random.randint(1, 10, n),
        "last_login_days_ago": np.random.randint(1, 120, n),
        "nps_score": np.random.randint(0, 11, n),
        "contract_type": np.random.choice(["Monthly", "Annual", "Multi-year"], n, p=[0.3, 0.5, 0.2]),
        "onboarding_completed": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "has_dedicated_csm": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "industry": np.random.choice(["Logistics", "HR-Tech", "Fintech", "Healthcare", "Retail"], n),
    }

    df = pd.DataFrame(data)

    # Churn logic (probabilistic based on observed SaaS patterns)
    # The higher the score, the higher the likelihood of churn.
    churn_score = (
        (df["tenure_months"] < 12).astype(int) * 2.0 +
        (df["support_tickets_last_90d"] > 10).astype(int) * 2.5 +
        (df["last_login_days_ago"] > 60).astype(int) * 4.0 +
        (df["nps_score"] < 6).astype(int) * 3.0 +
        (df["contract_type"] == "Monthly").astype(int) * 1.5 +
        (df["onboarding_completed"] == 0).astype(int) * 2.0 +
        (df["product_modules_used"] < 4).astype(int) * 1.5 +
        (df["has_dedicated_csm"] == 0).astype(int) * 1.0 +
        (df["num_users"] < 20).astype(int) * 1.0 +
        (df["monthly_spend"] < 2000).astype(int) * 1.0 +
        df["industry"].map({"Retail": 1.5, "Healthcare": 1.0, "Logistics": 0.5, "Fintech": 0.2, "HR-Tech": 0.0})
    )

    # Normalize to [0, 1] range based on theoretical max score (around 20)
    churn_prob = churn_score / 20.0
    
    # Add realistic noise so the relationship isn't perfectly linearly separable
    noise = np.random.normal(0, 0.1, n)
    churn_prob = np.clip(churn_prob + noise, 0, 1)

    # Probabilistic classification instead of hard threshold
    df["churned"] = np.random.binomial(1, churn_prob)

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_churn_dataset(1000)
    output_path = "data/clients.csv"
    df.to_csv(output_path, index=False)
    
    churn_rate = df['churned'].mean()
    print(f"Dataset generated: {len(df)} rows")
    print(f"Saved to: {output_path}")
    print(f"Overall Churn Rate: {churn_rate:.1%}")
