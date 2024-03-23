import argparse
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid

OPTS = None

def predict(w, X):
    """Return the predictions using weight vector w on inputs X.

    Args:
        - w: Vector of size (D,)
        - X: Matrix of size (M, D)
    Returns:
        - Predicted classes as a numpy vector of size (M,). Each entry should be either -1 or +1.
    """
    scores = np.matmul(X, w)
    probabilities = sigmoid(scores)
    preds = np.where(probabilities > 0.5, 1, -1)
    return preds

def train(X_train, y_train, lr=1e-1, num_iters=5000, l2_reg=0.0):
    """Train linear regression using gradient descent.

    Args:
        - X_train: Matrix of size (N, D)
        - y_train: Vector os size (N,)
        - lr: Learning rate
        - num_iters: Number of iterations of gradient descent to run
        - l2_reg: lambda hyperparameter for using L2 regularization
    Returns:
        - Weight vector w of size (D,)
    """
    N, D = X_train.shape
    w = np.zeros(D)
    for _ in tqdm(range(num_iters)):
        scores = np.matmul(X_train, w)
        preds = sigmoid(y_train * scores)
        gradient = -np.matmul((1 - preds) * y_train, X_train) / N
        gradient += l2_reg * w
        w-= lr * gradient
    return w

def evaluate(w, X, y, name):
    """Measure and print accuracy of a predictor on a dataset."""
    y_preds = predict(w, X)
    acc = np.mean(y_preds == y)
    print('    {} Accuracy: {}'.format(name, acc))
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=2)
    parser.add_argument('--num-iters', '-T', type=int, default=10000)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot-weights')
    return parser.parse_args()

def main():
    # Read the data
    path = '/mm_data'
    all_data = np.load('q1_data.npy', allow_pickle=True).item()
    X_train = all_data['X_train']
    y_train = all_data['y_train']
    X_dev = all_data['X_dev']
    y_dev = all_data['y_dev']
    X_test = all_data['X_test']
    y_test = all_data['y_test']

    # Train with gradient descent
    w = train(X_train, y_train, lr=OPTS.learning_rate, num_iters=OPTS.num_iters, l2_reg=OPTS.l2)

    # Evaluate model
    train_acc = evaluate(w, X_train, y_train, 'Train')
    dev_acc = evaluate(w, X_dev, y_dev, 'Dev')
    if OPTS.test:
        test_acc = evaluate(w, X_test, y_test, 'Test')

    # Plot the weights
    if OPTS.plot_weights:
        img = w.reshape(28, 28)
        vmax = max(np.max(w), -np.min(w))
        plt.imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.savefig(f'plot_{OPTS.plot_weights}.png')
        plt.show()


if __name__ == '__main__':
    OPTS = parse_args()
    main()