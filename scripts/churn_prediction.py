# churn_prediction_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Step 1: Load and Preprocess Data
# Load the dataset generated previously (saas_dataset.csv)
data = pd.read_csv("saas_dataset.csv")
data.dropna(inplace=True)  # Remove any missing values

# One-hot encode categorical columns: ContractType, SubscriptionPlan
data = pd.get_dummies(data, columns=['ContractType', 'SubscriptionPlan'], drop_first=True)

# Define features and target variable
features = ['Tenure', 'MonthlySpend', 'LoginFrequency', 'SupportTickets'] + \
           [col for col in data.columns if col.startswith('ContractType_') or col.startswith('SubscriptionPlan_')]
target = 'ChurnStatus'
X = data[features]
y = data[target]

# Step 2: Create Train-Test Split with stratification (important if classes are unbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Step 3: Build Preprocessing and Modeling Pipeline
# Separate numeric and categorical features for proper scaling
numeric_features = ['Tenure', 'MonthlySpend', 'LoginFrequency', 'SupportTickets']
categorical_features = [col for col in X.columns if col.startswith('ContractType_') or col.startswith('SubscriptionPlan_')]

# Preprocessing: Scale numeric features and pass-through categorical features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', 'passthrough', categorical_features)
])

# Build the pipeline with Logistic Regression (a simple, interpretable model)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))
])

# Step 4: Train the Model
pipeline.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("Model Evaluation Metrics:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred_proba))

# Step 6: Export Predictions for BI Integration
# Adding predictions to the test dataset helps inform targeted retention efforts.
predictions_df = X_test.copy()
predictions_df['Actual_Churn'] = y_test.values
predictions_df['Predicted_Churn'] = y_pred
predictions_df['Churn_Probability'] = y_pred_proba
predictions_df.to_csv("churn_predictions.csv", index=False)

print("Churn predictions, including probability estimates, have been exported to churn_predictions.csv.")
