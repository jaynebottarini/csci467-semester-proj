import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
import random

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop(columns=['OUTCOME', "TEAM", "ROUND", "BY YEAR NO", "BY ROUND NO", "SCORE"])
    y = df['OUTCOME']
    return X, y

def calculate_precision(y_pred, y_real):

    if len(y_pred) != len(y_real):
        raise ValueError("Lengths of y_pred and y_real must be the same.")

    true_positives = 0
    false_positives = 0

    for pred, real in zip(y_pred, y_real):
        if pred == 1 and real == 1:
            true_positives += 1
        elif pred == 1 and real == 0:
            false_positives += 1

    if true_positives + false_positives == 0:
        precision = 0  
    else:
        precision = true_positives / (true_positives + false_positives)

    return precision

def evaluate_model(X, y):
    i = 0
    y_pred = []
    while i < len(X):
        if X[i] < X[i+1]:
            y_pred.append('W')
            y_pred.append('L')
        elif X[i] > X[i+1]:
            y_pred.append('L')
            y_pred.append('W')
        else:
            prediction_order = random.sample(['L', 'W'], 1)
            y_pred.extend(prediction_order)
            if prediction_order == 'W':
                y_pred.append('L')
            else:
                y_pred.append('W')

        i = i + 2

    count_correct = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y[i]:
            count_correct = count_correct + 1

    count_correct_w = 0
    count_num_w = 0

    for i in range(len(y_pred)):
        if y[i] == 'W':
            count_num_w += 1
            if y_pred[i] == y[i]:
                count_correct_w = count_correct_w + 1

    count_correct_l = 0
    count_num_l = 0

    for i in range(len(y_pred)):
        if y[i] == 'L':
            count_num_l += 1
            if y_pred[i] == y[i]:
                count_correct_l = count_correct_l + 1

    precision_w = count_correct_w / count_num_w
    precision_l = count_correct_l / count_num_l
    
    print("accuracy:", count_correct / len(y_pred))
    f1 = f1_score(y, y_pred, average='weighted')
    print("f1-score:", f1)
    print("precision for win:", precision_w)
    print("precision for loss:", precision_l)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))



def main(file_path):
    matchups = load_data(file_path)

    X, y = preprocess_data(matchups)
    print(matchups["SEED"])

    evaluate_model(matchups["SEED"], y)



if __name__ == "__main__":
    file_path = "matchups.csv"
    main(file_path)