import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop(columns=['OUTCOME', "TEAM", "ROUND", "BY YEAR NO", "BY ROUND NO", "SCORE"])
    y = df['OUTCOME']
    print(X.head())
    return X, y

def evaluate_model(model, X_test, y_test, incorrect_sample_size=5, correct_sample_size=5):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Error Analysis
    incorrect_indices = (y_pred != y_test)
    incorrect_predictions = X_test[incorrect_indices]
    actual_labels = y_test[incorrect_indices]
    predicted_labels = y_pred[incorrect_indices]

    print("\nIncorrect Predictions:")
    for i, (instance, actual, predicted) in enumerate(zip(incorrect_predictions[:incorrect_sample_size], actual_labels[:incorrect_sample_size], predicted_labels[:incorrect_sample_size])):
        print(f"Instance {i+1}:")
        print("Features:", instance)
        print("Actual Label:", actual)
        print("Predicted Label:", predicted)
        print("-------------")

    # Random correct predictions
    correct_indices = (y_pred == y_test)
    correct_predictions = X_test[correct_indices]
    actual_labels_correct = y_test[correct_indices]
    predicted_labels_correct = y_pred[correct_indices]

    print("\nRandom Correct Predictions:")
    if len(correct_predictions) >= correct_sample_size:
        random_correct_indices = np.random.choice(len(correct_predictions), correct_sample_size, replace=False)
        for i, index in enumerate(random_correct_indices):
            instance = correct_predictions[index]
            actual = actual_labels_correct.iloc[index]  # Use iloc to access by integer position
            predicted = predicted_labels_correct[index]
            print(f"Instance {i+1}:")
            print("Features:", instance)
            print("Actual Label:", actual)
            print("Predicted Label:", predicted)
            print("-------------")
    else:
        print("Not enough correct predictions to sample.")


def main(file_path):
    matchups = load_data(file_path)

    X, y = preprocess_data(matchups)

    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {'C': [0, 0.0001, 0.001, 0.01, 0.1]}  
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_dev_scaled, y_dev)
    best_params = grid_search.best_params_

    print("best parameters:", best_params)

    model = LogisticRegression(**best_params)
    model.fit(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)



if __name__ == "__main__":
    file_path = "matchups.csv"
    main(file_path)