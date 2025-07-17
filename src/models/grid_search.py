import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# read data
X_train = pd.read_csv('data/scaled_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

# convert y_train to 1D array (scikit-learn expects 1D array, not column vector)
y_train = y_train.iloc[:, 0]  # or y_train.values.ravel()

# define model & grid search
model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)

# fit model
grid_search.fit(X_train, y_train)

# save model parameters
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')


print("Grid search completed successfully 🎉")