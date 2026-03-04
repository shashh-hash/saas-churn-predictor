import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

def generate_churn_dataset(n=1000):
    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "tenure_months": np.random.randint(1, 60, n),
        "monthly_spend": np.round(np.random.uniform(500, 10000, n), 2),
        "num_users": np.random.randint(5, 200, n),
        "support_tickets_last_90d": np.random.randint(0, 20, n),
        "product_modules_used": np.random.randint(1, 8, n),
        "last_login_days_ago": np.random.randint(1, 120, n),
        "nps_score": np.random.randint(0, 10, n),
        "contract_type": np.random.choice(["Monthly", "Annual", "Multi-year"], n, p=[0.3, 0.5, 0.2]),
        "onboarding_completed": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "has_dedicated_csm": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "industry": np.random.choice(["Logistics", "HR-Tech", "Fintech", "Healthcare", "Retail"], n),
    }

    df = pd.DataFrame(data)

    # Churn logic (realistic based on SaaS patterns)
    churn_score = (
        (df["tenure_months"] < 6).astype(int) * 2 +
        (df["support_tickets_last_90d"] > 10).astype(int) * 2 +
        (df["last_login_days_ago"] > 60).astype(int) * 3 +
        (df["nps_score"] < 5).astype(int) * 2 +
        (df["contract_type"] == "Monthly").astype(int) * 1 +
        (df["onboarding_completed"] == 0).astype(int) * 2 +
        (df["product_modules_used"] < 3).astype(int) * 1 +
        (df["has_dedicated_csm"] == 0).astype(int) * 1
    )

    churn_prob = churn_score / churn_score.max()
    df["churned"] = (churn_prob > np.random.uniform(0.3, 0.7, n)).astype(int)

    return df

if __name__ == "__main__":
    df = generate_churn_dataset()
    df.to_csv("data.csv", index=False)
    print(f"Dataset generated: {len(df)} rows, {df['churned'].mean():.1%} churn rate")
