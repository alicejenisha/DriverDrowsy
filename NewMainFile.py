import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the path to your CSV file
file_path = 'Dataset/dataset (1).csv'

# Read the data from the CSV file
data = pd.read_csv(file_path)

# Display the first 20 rows of the data
print("First 20 rows of the data:")
print(data.head(20))

# Check for missing values and duplicates
print("\n--------------  DATA PREPROCESSING  --------------\n")
print("Check for missing values:")
print(data.isnull().sum())
print("Check for duplicate values:")
print(data.duplicated().sum())

# Label Encoding
print("\n--------------  LABEL ENCODING  --------------\n")
data['type'] = pd.factorize(data['type'])[0]
data['nameOrig'] = pd.factorize(data['nameOrig'])[0]
data['nameDest'] = pd.factorize(data['nameDest'])[0]

# Data after Label Encoding
print("\nData after Label Encoding:")
print(data.head(20))

# Drop 'isFlaggedFraud' column if it exists
if 'isFlaggedFraud' in data.columns:
    data = data.drop(columns=['isFlaggedFraud'])

# Data Splitting
X = data.drop(columns='isFraud')  # Features
Y = data['isFraud']  # Target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the data shapes
print("\n--------------  DATA SIZE'S  --------------\n")
print(f"The shape of the data: {data.shape}")
print(f"Training set X shape: {X_train.shape}, Y shape: {Y_train.shape}")
print(f"Testing set X shape: {X_test.shape}, Y shape: {Y_test.shape}")
print(" \n--------------  ALGORITHMS   --------------\n")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(" \n--------------  RANDOM FOREST CLASSIFIER   --------------\n")

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Predict on the test set
Y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
print(f"\nAccuracy of Random Forest Classifier: {accuracy_rf*100:.4f}%")

# Random Forest Classification Report and Confusion Matrix
print("\nClassification Report for Random Forest:")
print(classification_report(Y_test, Y_pred_rf))

# print("\nConfusion Matrix for Random Forest:")
# conf_matrix_rf = confusion_matrix(Y_test, Y_pred_rf)
# sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title("Confusion Matrix for Random Forest")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

print(" \n--------------  PLOT CONFUSION MATRIX   --------------\n")

# Compute confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_rf)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix for Random Forest Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Feature importance (optional)
print("\nFeature Importances:")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))


import xgboost as xgb

print(" \n--------------  XGBOOST CLASSIFIER   --------------\n")

# Train the XGBoost Classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, Y_train)

# Predict on the test set
Y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(Y_test, Y_pred_xgb)
print(f"\nAccuracy of XGBoost Classifier: {accuracy_xgb*100:.4f}%")

# XGBoost Classification Report and Confusion Matrix
print("\nClassification Report for XGBoost:")
print(classification_report(Y_test, Y_pred_xgb))

# print("\nConfusion Matrix for XGBoost:")
# conf_matrix_xgb = confusion_matrix(Y_test, Y_pred_xgb)
# sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title("Confusion Matrix for XGBoost")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()
print(" \n--------------  PLOT CONFUSION MATRIX FOR XGBOOST   --------------\n")

# Plotting the confusion matrix for XGBoost
conf_matrix_xgb = confusion_matrix(Y_test, Y_pred_xgb)


# Plotting the confusion matrix for XGBoost with blue color
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix for XGBoost Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
# Get user input for a new transaction
print("\n--------------  ENTER DETAILS FOR A NEW TRANSACTION  --------------")

# Collect user input for a new transaction
transaction_data = {
    'step': float(input("Step: ")),
    'type': int(input("Type (Encoded, e.g., 0, 1, 2): ")),
    'amount': float(input("Amount: ")),
    'nameOrig': int(input("nameOrig (Encoded): ")),
    'oldbalanceOrg': float(input("oldbalanceOrg: ")),
    'newbalanceOrig': float(input("newbalanceOrig: ")),
    'nameDest': int(input("nameDest (Encoded): ")),
    'oldbalanceDest': float(input("oldbalanceDest: ")),
    'newbalanceDest': float(input("newbalanceDest: "))
}

# Convert the input data into a DataFrame
new_transaction = pd.DataFrame([transaction_data])

# Make sure the new input matches the training set columns
new_transaction = new_transaction[X.columns]

# Predict using Random Forest
rf_prediction = rf_model.predict(new_transaction)
print(f"\nRandom Forest Prediction for isFraud: {'Fraud' if rf_prediction[0] == 1 else 'Not Fraud'}")

# Predict using XGBoost
xgb_prediction = xgb_model.predict(new_transaction)
print(f"XGBoost Prediction for isFraud: {'Fraud' if xgb_prediction[0] == 1 else 'Not Fraud'}")




import joblib

# Save the Random Forest model to a .pkl file
joblib.dump(rf_model, 'random_forest_model.pkl')

# Save the XGBoost model to a .pkl file
joblib.dump(xgb_model, 'xgboost_model.pkl')

print("Models saved successfully as .pkl files.")
import pickle

# Save the Random Forest model to a .pkl file
with open('random_forest_model.pkl', 'wb') as rf_file:
    pickle.dump(rf_model, rf_file)

# Save the XGBoost model to a .pkl file
with open('xgboost_model.pkl', 'wb') as xgb_file:
    pickle.dump(xgb_model, xgb_file)

print("Models saved successfully as .pkl files.")

# Load the Random Forest model
rf_model_loaded = joblib.load('random_forest_model.pkl')

# Load the XGBoost model
xgb_model_loaded = joblib.load('xgboost_model.pkl')

print("Models loaded successfully.")
# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as rf_file:
    rf_model_loaded = pickle.load(rf_file)

# Load the XGBoost model
with open('xgboost_model.pkl', 'rb') as xgb_file:
    xgb_model_loaded = pickle.load(xgb_file)

print("Models loaded successfully.")
from playsound import playsound
# If any model predicts Fraud, play the alert sound
if rf_prediction[0] == 1 or xgb_prediction[0] == 1:
    print("\nALERT: Fraud detected!")
    playsound('siren-alert-96052.mp3')