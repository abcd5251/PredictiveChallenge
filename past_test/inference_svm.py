import pandas as pd
import numpy as np
from sklearn.svm import SVR
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression


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


# Train Support Vector Regressor (SVM)
def train_svm(df, selector_path):
    # Define features (X) and label (Y)
    feature_columns = [col for col in df.columns if col not in ['id', 'project_a', 'project_b', 'weight_a', 'weight_b']]
    X = df[feature_columns]
    Y = df['weight_a']

    # Select the top 5 features based on univariate statistical tests
    selector = SelectKBest(score_func=f_regression, k=6)
    X_new = selector.fit_transform(X, Y)

    # Save the selector for later use
    with open(selector_path, 'wb') as file:
        pickle.dump(selector, file)

    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selector.get_support()[i]]

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=42)

    # Create and train the SVM model
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, Y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    print(f"RMSE: {rmse}")
    print(f"Selected Features: {selected_features}")

    return model


# Save the trained model
def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


# Predict using the saved model
def predict_with_model(model_path, selector_path, test_csv_path, output_csv_path):
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load the feature selector
    with open(selector_path, 'rb') as file:
        selector = pickle.load(file)

    # Load and preprocess the test data
    test_data = load_data(test_csv_path)
    processed_test_data = preprocess_data(test_data)

    # Extract features for prediction
    feature_columns = [col for col in processed_test_data.columns if col not in ['id', 'project_a', 'project_b', 'weight_a', 'weight_b']]
    X_test = processed_test_data[feature_columns]

    # Transform the test data using the saved selector
    X_test_new = selector.transform(X_test)

    # Make predictions
    predictions = model.predict(X_test_new)

    # Adjust predictions based on the conditions
    predictions = np.where(predictions >= 1, 0.9, np.where(predictions <= 0, 0.1, predictions))

    # Save results to CSV
    result = pd.DataFrame({"id": test_data["id"], "pred": predictions})
    result.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


# Main script
if __name__ == "__main__":
    # File paths
    train_csv_path = "train_data.csv"
    test_csv_path = "test_data.csv"
    model_save_path = "svm_model.pkl"
    selector_save_path = "feature_selector.pkl"
    result_csv_path = "result.csv"

    # Load and preprocess the training data
    data = load_data(train_csv_path)
    processed_data = preprocess_data(data)

    # Train the SVM model
    svm_model = train_svm(processed_data, selector_save_path)

    # Save the model
    save_model(svm_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Predict on test data and save results
    predict_with_model(model_save_path, selector_save_path, test_csv_path, result_csv_path)
