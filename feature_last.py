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
def preprocess_data(df, is_training=True):
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

    # Keep only the necessary columns
    columns_to_keep = ['project_a_star_count', 'project_b_star_count']
    if is_training:
        columns_to_keep.append('weight_a')

    df = df[columns_to_keep]

    # Fill missing values with the mean of each column
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

    return df

# Train XGBoost regressor
def train_xgboost(df):
    # Define features (X) and label (Y)
    feature_columns = ['project_a_star_count', 'project_b_star_count']
    X = df[feature_columns]
    Y = df['weight_a']

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("Length of test set:", len(X_test))

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
    processed_test_data = preprocess_data(test_data, is_training=False)

    # Extract features for prediction
    feature_columns = ['project_a_star_count', 'project_b_star_count']
    X_test = processed_test_data[feature_columns]

    # Make predictions
    predictions = model.predict(X_test)

    # Adjust predictions based on the conditions
    predictions = np.where(predictions >= 1, 0.98, np.where(predictions <= 0, 0.02, predictions))

    # Save results to CSV
    result = pd.DataFrame({"id": test_data.index, "pred": predictions})
    result.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# Main script
if __name__ == "__main__":
    # File paths
    train_csv_path = "train_data.csv"
    test_csv_path = "test_data.csv"
    model_save_path = "xgboost_model.pkl"
    result_csv_path = "aa_result.csv"

    # Load and preprocess the training data
    data = load_data(train_csv_path)
    processed_data = preprocess_data(data, is_training=True)

    # Train the XGBoost model
    xgboost_model = train_xgboost(processed_data)

    # Save the model
    save_model(xgboost_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Predict on test data and save results
    predict_with_model(model_save_path, test_csv_path, result_csv_path)
