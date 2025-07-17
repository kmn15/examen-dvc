import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# read data
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

# drop date column
if 'date' in X_train.columns:
    X_train = X_train.drop('date', axis=1)
if 'date' in X_test.columns:
    X_test = X_test.drop('date', axis=1)

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# save scaled data
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/scaled_data/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/scaled_data/X_test_scaled.csv', index=False)

# save scaler
joblib.dump(scaler, 'models/scaler.joblib')

print("Data scaled successfully 🎉")