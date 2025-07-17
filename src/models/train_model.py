import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# read data
X_train = pd.read_csv('data/scaled_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()

# read model parameters
best_params = joblib.load('models/best_params.pkl')

# define model
model = RandomForestRegressor(**best_params, random_state=42)

# fit model
model.fit(X_train, y_train)

# save model
joblib.dump(model, 'models/best_model.pkl')

print("Model trained successfully 🎉")