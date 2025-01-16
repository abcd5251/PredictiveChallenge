import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the train.csv file
def load_data(file_path):
    return pd.read_csv(file_path)

# Convert strings with 'k' to numerical values, handle missing values, and calculate column averages
def preprocess_data(df):
    def convert_to_int(value):
        if isinstance(value, str):
            if 'k' in value:
                try:
                    return int(float(value.replace('k', '')) * 1000)
                except ValueError:
                    return np.nan
            try:
                return int(value.replace(",", ""))
            except ValueError:
                return np.nan
        return value

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(convert_to_int)

    # Fill missing values with the mean of each column
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    return df

# Train XGBoost regressor
def train_xgboost(df):
    # Define features (X) and label (Y)
    feature_columns = [col for col in df.columns if col not in ['id', 'project_a', 'project_b', 'weight_a', 'weight_b']]
    X = df[feature_columns]
    Y = df['weight_a']

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, Y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    print(f"RMSE: {rmse}")

    return model

# Train XGBoost with top 3 important features
def train_with_top_features(df):
    # Define features (X) and label (Y)
    feature_columns = [col for col in df.columns if col not in ['id', 'project_a', 'project_b', 'weight_a', 'weight_b']]
    X = df[feature_columns]
    Y = df['weight_a']

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initial model to calculate feature importance
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, Y_train)

    # Get feature importance and select top 3 features
    feature_importances = model.feature_importances_
    top_feature_indices = np.argsort(feature_importances)[-8:]
    top_features = [feature_columns[i] for i in top_feature_indices]

    # Train a new model with only top 3 features
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    top_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    top_model.fit(X_train_top, Y_train)

    # Make predictions and evaluate the model
    predictions = top_model.predict(X_test_top)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    print(f"RMSE with top features: {rmse}")

    return top_model, top_features

# Save the trained model along with features
def save_model_with_features(model, file_path, features=None):
    with open(file_path, 'wb') as file:
        pickle.dump({"model": model, "features": features}, file)

# Predict using the saved model
def predict_with_model(model_path, test_csv_path, output_csv_path):
    # Load the model and features
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
        model = data["model"]
        features = data.get("features")

    # Load and preprocess the test data
    test_data = load_data(test_csv_path)
    processed_test_data = preprocess_data(test_data)

    # Align features for prediction
    if features:
        X_test = processed_test_data[features]
    else:
        feature_columns = [col for col in processed_test_data.columns if col not in ['id', 'project_a', 'project_b']]
        X_test = processed_test_data[feature_columns]

    # Make predictions
    predictions = model.predict(X_test)

    # Adjust predictions based on the conditions
    predictions = np.where(predictions >= 1, 0.92, np.where(predictions <= 0, 0.08, predictions))

    # Save results to CSV
    result = pd.DataFrame({"id": test_data["id"], "pred": predictions})
    result.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# Main script
if __name__ == "__main__":
    # File paths
    train_csv_path = "train_data.csv"
    test_csv_path = "test_data.csv"
    model_save_path = "xgboost_model.pkl"
    top_model_save_path = "xgboost_top_model.pkl"
    result_csv_path = "result.csv"
    top_result_csv_path = "result_top.csv"

    # Load and preprocess the training data
    data = load_data(train_csv_path)
    processed_data = preprocess_data(data)

    # Train the XGBoost model
    xgboost_model = train_xgboost(processed_data)

    # Save the model
    save_model_with_features(xgboost_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Train with top features
    top_model, top_features = train_with_top_features(processed_data)
    print(f"Top features: {top_features}")

    # Save the top feature model
    save_model_with_features(top_model, top_model_save_path, features=top_features)
    print(f"Top feature model saved to {top_model_save_path}")

    # Predict on test data and save results
    predict_with_model(model_save_path, test_csv_path, result_csv_path)

    # Predict with top feature model
    predict_with_model(top_model_save_path, test_csv_path, top_result_csv_path)
