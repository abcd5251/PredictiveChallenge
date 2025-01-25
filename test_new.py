import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

top_feature = 20

# Load the train.csv file
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess the dataset by dropping unnecessary columns and handling missing values
def preprocess_data(df):

    def convert_to_float(value):
        if isinstance(value, str):
            if 'k' in value:
                try:
                    return float(value.replace('k', '')) * 1000
                except ValueError:
                    return np.nan
            try:
                return float(value.replace(",", ""))
            except ValueError:
                return np.nan
        return value

    # Convert relevant columns to floats
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(convert_to_float)

    # Convert last commit time to a numeric feature (days since last commit)
    current_date = datetime.now()
    date_columns = ['project_a_repo_last_commit_time', 'project_b_repo_last_commit_time']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = (current_date - df[col]).dt.days

    # Choose contributors column with the most non-null values
    df['project_a_contributors_count'] = df[['project_a_contributors_count', 'project_a_repo_contributors_to_repo_count']].max(axis=1)
    df['project_b_contributors_count'] = df[['project_b_contributors_count', 'project_b_repo_contributors_to_repo_count']].max(axis=1)

    # Drop unnecessary columns
    columns_to_drop = [
        'project_a', 'project_b',
        'project_a_repo_language', 'project_b_repo_language',
        'project_a_repo_license_spdx_id', 'project_b_repo_license_spdx_id',
        'project_a_repo_image_path', 'project_b_repo_image_path',
        'project_a_repo_description', 'project_b_repo_description',
        'project_a_repo_created_at', 'project_b_repo_created_at',
        'project_a_repo_updated_at', 'project_b_repo_updated_at',
        'project_a_repo_first_commit_time', 'project_b_repo_first_commit_time'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Fill missing values with the mean of each column
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    return df

# Train XGBoost regressor and return top 10 features
def train_xgboost(df):
    # Define features (X) and label (Y)
    feature_columns = [col for col in df.columns if col not in ['id', 'project_a', 'project_b', 'weight_a', 'weight_b']]
    X = df[feature_columns]
    Y = df['weight_a']

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.000001, random_state=42)

    # Create and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=400, learning_rate=0.12, max_depth=4, random_state=888)
    model.fit(X_train, Y_train)

    # Get feature importance
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_columns, 'importance': feature_importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False).head(top_feature)
    top_features = importance_df['feature'].tolist()
    print(top_features)
    # Validate model with top 10 features
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    model.fit(X_train_top, Y_train)
    predictions = model.predict(X_test_top)
    predictions = np.where(predictions >= 1, 0.99, np.where(predictions <= 0, 0.01, predictions))
    mse = mean_squared_error(Y_test, predictions)
    print(f"MSE with top 10 features: {mse}")

    return model, top_features

# Save the trained model
def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

# Predict using the saved model
def predict_with_model(model_path, test_csv_path, output_csv_path, top_features):
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load and preprocess the test data
    test_data = load_data(test_csv_path)
    processed_test_data = preprocess_data(test_data)

    # Extract top features for prediction
    X_test = processed_test_data[top_features]

    # Make predictions
    predictions = model.predict(X_test)

    # Adjust predictions based on the conditions
    predictions = np.where(predictions >= 1, 0.99, np.where(predictions <= 0, 0.01, predictions))

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
    result_csv_path = "new_tune_result.csv"

    # Load and preprocess the training data
    data = load_data(train_csv_path)
    processed_data = preprocess_data(data)

    # Train the XGBoost model and get top features
    xgboost_model, top_features = train_xgboost(processed_data)

    # Save the model
    save_model(xgboost_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Predict on test data with top features and save results
    predict_with_model(model_save_path, test_csv_path, result_csv_path, top_features)
