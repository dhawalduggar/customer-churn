import pandas as pd
from sqlalchemy import create_engine

# Load the original dataset and predictions
original_data = pd.read_csv("saas_dataset.csv")
predictions = pd.read_csv("churn_predictions.csv")

# Merge predictions with the original dataset for enriched analysis
merged_data = pd.merge(original_data, predictions, left_index=True, right_index=True)

# Save to CSV for Power BI import
merged_data.to_csv("churn_data_for_bi.csv", index=False)
print("Data exported to churn_data_for_bi.csv for Power BI integration.")