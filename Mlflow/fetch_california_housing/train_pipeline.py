from sklearn.datasets import fetch_california_housing
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_and_processing_data():
    df = fetch_california_housing(as_frame=True).frame
    return df

def split_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
        
    n_estimators = [100, 150, 200]
    max_depth = [3, 4, 5, 6, 7]

    for n in n_estimators:
        for d in max_depth:
            grad_model = GradientBoostingRegressor(n_estimators=n, 
                                                    max_depth=d, 
                                                    learning_rate=0.1, 
                                                    loss='squared_error',
                                                    random_state=42)

            forest_model = RandomForestRegressor(n_estimators=n, 
                                                    max_depth=d,
                                                    random_state=42)
            
            grad_model.fit(X_train, y_train)
            y_pred = grad_model.predict(X_test)
            
            forest_model.fit(X_train, y_train)
            forest_pred = forest_model.predict(X_test)
            
            grad_mse = mean_squared_error(y_test, y_pred)
            grad_r2 = r2_score(y_test, y_pred)

            forest_mse = mean_squared_error(y_test, forest_pred)
            forest_r2 = r2_score(y_test, forest_pred)
            print(f"========= N_ESTIMATORS: {n}, MAX_DEPTH: {d} ===========")
            print(f"Gradient Boosting Regression: ")
            print(f"  MSE: {grad_mse}")
            print(f"  R2: {grad_r2}")
            print(f"Random Forest Regression: ")
            print(f"  MSE: {forest_mse}")
            print(f"  R2: {forest_r2}")
    print("=================== Check model evaluation ===================")
    print("Model training completed successfully")