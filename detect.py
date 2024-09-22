import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# Example new transaction data (replace with actual values)
new_transaction = {
    'Time': [20000],  # example timestamp
    'V1': [-1.3598],  # feature values from V1 to V28 (replace with actual data)
    'V2': [-0.0727],
    'V3': [2.5363],
    'V4': [1.3782],
    'V5': [-0.3383],
    'V6': [0.4624],
    'V7': [0.2396],
    'V8': [0.0987],
    'V9': [0.3639],
    'V10': [0.0902],
    'V11': [-0.5516],
    'V12': [0.0214],
    'V13': [-0.5281],
    'V14': [-0.0367],
    'V15': [0.2395],
    'V16': [0.0987],
    'V17': [0.3639],
    'V18': [0.0902],
    'V19': [-0.5516],
    'V20': [0.0214],
    'V21': [-0.5281],
    'V22': [-0.0367],
    'V23': [0.2395],
    'V24': [0.0987],
    'V25': [0.3639],
    'V26': [0.0902],
    'V27': [-0.5516],
    'V28': [0.0214],
    'Amount': [50]  # transaction amount
}

# Convert to DataFrame
new_transaction_df = pd.DataFrame(new_transaction)

# Preprocess the new transaction (scale it like the training data)
scaler = StandardScaler()
new_transaction_scaled = scaler.fit_transform(new_transaction_df)

# Predict using the loaded model
prediction = model.predict(new_transaction_scaled)

# Output prediction
if prediction[0] == 1:
    print("Fraudulent Transaction Detected")
else:
    print("Non-Fraudulent Transaction")
