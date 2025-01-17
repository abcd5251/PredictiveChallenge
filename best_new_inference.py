import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the train.csv file
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess the dataset by dropping unnecessary columns and handling missing values
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

    # Convert relevant columns to integers
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(convert_to_int)

    # Choose contributors column with the most non-null values
    df['project_a_contributors_count'] = df[['project_a_contributors_count', 'project_a_repo_contributors_to_repo_count']].max(axis=1)
    df['project_b_contributors_count'] = df[['project_b_contributors_count', 'project_b_repo_contributors_to_repo_count']].max(axis=1)

    # Drop unnecessary columns
    columns_to_drop = [
        'project_a_star_count', 'project_a_fork_count', 'project_a_watcher_count',
        'project_b_star_count', 'project_b_fork_count', 'project_b_watcher_count',
        'project_a_repo_contributors_to_repo_count', 'project_b_repo_contributors_to_repo_count',
        'project_a_repo_image_path', 'project_b_repo_image_path',
        'project_a_repo_description', 'project_b_repo_description',
        'project_a_repo_last_commit_time', 'project_b_repo_last_commit_time',
        'project_a_repo_language', 'project_b_repo_language',
        'project_a_repo_license_spdx_id', 'project_b_repo_license_spdx_id',
        'project_a_repo_created_at', 'project_a_repo_updated_at',
        'project_a_repo_first_commit_time', 'project_b_repo_created_at',
        'project_b_repo_updated_at', 'project_b_repo_first_commit_time',
        'project_b_community_engagement', 'project_b_accessibility',
        'project_b_Readme_score', 'project_b_technical_innovation',
        'project_a_community_engagement', 'project_a_accessibility',
        'project_a_Readme_score', 'project_a_technical_innovation'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.0001, random_state=42)
    print("length of test", len(X_test))
    # Create and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, Y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    predictions = np.where(predictions >= 1, 0.98, np.where(predictions <= 0, 0.02, predictions))
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    print(f"RMSE: {rmse}")

    return model

# Save the trained model
def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

# Predict using the saved model
def predict_with_model(model_path, test_csv_path, output_csv_path):
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load and preprocess the test data
    test_data = load_data(test_csv_path)
    processed_test_data = preprocess_data(test_data)

    # Extract features for prediction
    feature_columns = [col for col in processed_test_data.columns if col not in ['id', 'project_a', 'project_b']]
    X_test = processed_test_data[feature_columns]

    # Make predictions
    predictions = model.predict(X_test)

    # Adjust predictions based on the conditions
    predictions = np.where(predictions >= 1, 0.98, np.where(predictions <= 0, 0.02, predictions))

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
    result_csv_path = "new_result.csv"

    # Load and preprocess the training data
    data = load_data(train_csv_path)
    processed_data = preprocess_data(data)

    # Train the XGBoost model
    xgboost_model = train_xgboost(processed_data)

    # Save the model
    save_model(xgboost_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Predict on test data and save results
    predict_with_model(model_save_path, test_csv_path, result_csv_path)
