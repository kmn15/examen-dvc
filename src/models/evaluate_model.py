import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

# read data
X_test = pd.read_csv('data/scaled_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()

# read model
model = joblib.load('models/best_model.pkl')

# make predictions
y_pred = model.predict(X_test)

# calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# save metrics
with open('metrics/scores.json', 'w') as f:
    json.dump({'mse': mse, 'mae': mae, 'r2': r2}, f)

# save predictions
predictions_df = pd.DataFrame({
    'y_test': y_test, 
    'y_pred': y_pred
})
predictions_df.to_csv('data/predictions/predictions.csv', index=False)

print("Model evaluated successfully 🎉")