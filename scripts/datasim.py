# datasim.py - Revised with realistic patterns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def generate_realistic_data():
    # Generate base features with statistical patterns
    n_customers = 5000
    telecom_data = pd.DataFrame({
        'CustomerID': [f"CUST_{i}" for i in range(n_customers)],
        'Tenure': np.random.lognormal(mean=2.5, sigma=0.3, size=n_customers).astype(int),
        'MonthlySpend': np.random.gamma(shape=2, scale=25, size=n_customers),
        'ContractType': np.random.choice(['Monthly', 'Yearly', 'Two-Year'], size=n_customers, p=[0.4, 0.4, 0.2])
    })
    
    # Cluster-based subscription plans
    kmeans = KMeans(n_clusters=3)
    telecom_data['SubscriptionPlan'] = kmeans.fit_predict(telecom_data[['MonthlySpend']])
    plan_mapping = {0: 'Basic', 1: 'Premium', 2: 'Enterprise'}
    telecom_data['SubscriptionPlan'] = telecom_data['SubscriptionPlan'].map(plan_mapping)
    
    # Realistic login patterns
    telecom_data['LoginFrequency'] = telecom_data.apply(
        lambda x: np.random.poisson(lam=(x['MonthlySpend']/10 + 5)), axis=1)
    
    # Support tickets with temporal decay
    telecom_data['SupportTickets'] = np.where(
        telecom_data['Tenure'] < 6,
        np.random.poisson(lam=3),
        np.random.poisson(lam=1)
    )
    
    # Survival analysis-based churn
    base_churn = 0.2
    telecom_data['ChurnRisk'] = base_churn + (
        0.001 * telecom_data['Tenure'] -
        0.005 * telecom_data['MonthlySpend'] +
        0.02 * telecom_data['SupportTickets']
    )
    telecom_data['ChurnStatus'] = np.where(telecom_data['ChurnRisk'] > np.random.rand(len(telecom_data)), 1, 0)
    
    return telecom_data

if __name__ == "__main__":
    data = generate_realistic_data()
    data.to_csv("saas_dataset.csv", index=False)
    print("Dataset generated with realistic patterns")
