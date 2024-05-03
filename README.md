## Dataset

The data for this project was sourced from Kaggle prior to the 2024 tournament, so experiments only include data from 2008 - 2023, excluding 2020.
https://www.kaggle.com/datasets/nishaanamin/march-madness-data

## Restructured Data

All data used for experiments was compiled and restructured in data_sorting.ipynb and outputted to files matchups.csv and matchups_combined.csv, where matchups is normalized and the binary classification task between winning and losing and matchups_combined is not normalized and the binary classification task between an upset or no upset.

## Baseline

To run the baseline, use the following command in terminal:
    python baseline.py

## Logistic Regression

To run the logistic regression experiment, use the command in terminal:
    python logistic_reg.py

## SVM

To run the SVM on the win/loss structure, run all cells in svm.ipynb
To run the SVM on the upset dataset, run all cells in svm_combined.ipynb

## Neural Network

To run the neural network on the win/loss structure, run all cells in nn.ipynb
To run the neural network on the upset dataset, run all cells in nn_combined.ipynb