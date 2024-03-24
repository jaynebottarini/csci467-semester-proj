import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop(columns=['OUTCOME', "TEAM", "ROUND", "BY YEAR NO", "BY ROUND NO", "SCORE"])
    y = df['OUTCOME']
    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main(file_path):
    # Load the dataset
    matchups = load_data(file_path)

    # Preprocess the data
    X, y = preprocess_data(matchups)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize logistic regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    file_path = "matchups.csv"
    main(file_path)