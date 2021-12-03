import csv

import matplotlib.pyplot as plt

from utils_age import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, alpha, beta):
    """ Compute the negative log-likelihood.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param alpha: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    for i in range(len(data["user_id"])):
        correct = data["is_correct"][i]
        user_key = data["user_id"][i]
        question_key = data["question_id"][i]
        # from the equation derived in a)
        factor = alpha[question_key] * (theta[user_key] - beta[question_key])
        log_lklihood += correct * factor - np.log(1 + np.exp(factor))
    return -log_lklihood


def update_theta_beta(data, lr, theta, alpha, beta):
    """ Update theta and beta using gradient descent.
    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param alpha: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    theta_array = np.array(theta[data["user_id"]])
    beta_array = np.array(beta[data["question_id"]])
    alpha_array = np.array(alpha[data["question_id"]])
    diff = (theta_array - beta_array)
    likli = sigmoid(alpha_array * diff)
    derivative_factor = (data["is_correct"] - likli)
    derivative_theta = alpha_array * derivative_factor
    derivative_alpha = diff * derivative_factor
    derivative_beta = -derivative_theta
    # likeli is a large array needs grouping use bincount
    theta += lr * np.bincount(data["user_id"], derivative_theta)
    beta += lr * np.bincount(data["question_id"], derivative_beta)
    alpha += lr * np.bincount(data["question_id"], derivative_alpha)
    return theta, alpha, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    rows = max(data["user_id"]) + 1
    # rows = 542
    columns = max(data["question_id"]) + 1
    # columns = 1774
    # use random to set
    np.random.seed(1)
    theta = np.random.rand(rows)
    beta = np.random.rand(columns)
    alpha = np.random.rand(columns)
    train_likelihood = []
    valid_likelihood = []
    val_acc_lst = []
    train_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, alpha=alpha, beta=beta)
        train_likelihood.append(neg_lld)
        valid_lld = neg_log_likelihood(val_data, theta, alpha, beta)
        valid_likelihood.append(valid_lld)
        score_train = evaluate(data, theta, alpha, beta)
        train_acc_lst.append(score_train)
        score = evaluate(data=val_data, theta=theta, alpha=alpha, beta=beta)
        val_acc_lst.append(score)
        print("On training data set, NLLK: {} \t Score: {}".format(neg_lld,
                                                                   score))
        theta, alpha, beta = update_theta_beta(data, lr, theta, alpha, beta)

    return theta, alpha, beta, train_likelihood, valid_likelihood, train_acc_lst, val_acc_lst


def evaluate(data, theta, alpha, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param alpha: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data_elder, train_data_younger, train_data_unknown = load_train_csv("../data")
    val_data_elder, val_data_younger, val_data_unknown = load_valid_csv("../data")
    test_data_elder, test_data_younger, test_data_unknown = load_public_test_csv("../data")

    # set hyperparameter
    lr, iterations = 0.01, 25
    lst_iterations = [i for i in range(1, 26)]
    theta, alpha, beta, train_likelihoods, valid_likelihoods, train_acc_lst, \
    valid_acc_lst = irt(train_data_elder, val_data_elder, lr, iterations)
    plt.plot(lst_iterations, train_likelihoods, label="training set likelihood")
    plt.plot(lst_iterations, valid_likelihoods, label="validation set likelihood")
    plt.legend()
    plt.ylabel("Likelihoods")
    plt.xlabel("Iterations")
    plt.show()
    # part c)
    print("Final accuracy on the validation data is:" + str(evaluate(val_data_elder,
                                                                     theta, alpha, beta)))
    print("Final accuracy on the training data is:" + str(evaluate(test_data_elder,
                                                                   theta, alpha, beta)))

    theta, alpha, beta, train_likelihoods, valid_likelihoods, train_acc_lst, \
    valid_acc_lst = irt(train_data_younger, val_data_younger, lr, iterations)
    plt.plot(lst_iterations, train_likelihoods, label="training set likelihood")
    plt.plot(lst_iterations, valid_likelihoods,
             label="validation set likelihood")
    plt.legend()
    plt.ylabel("Likelihoods")
    plt.xlabel("Iterations")
    plt.show()
    # part c)
    print("Final accuracy on the validation data is:" + str(
        evaluate(val_data_younger,
                 theta, alpha, beta)))
    print("Final accuracy on the training data is:" + str(
        evaluate(test_data_younger,
                 theta, alpha, beta)))
    theta, alpha, beta, train_likelihoods, valid_likelihoods, train_acc_lst, \
    valid_acc_lst = irt(train_data_unknown, val_data_unknown, lr, iterations)
    plt.plot(lst_iterations, train_likelihoods, label="training set likelihood")
    plt.plot(lst_iterations, valid_likelihoods,
             label="validation set likelihood")
    plt.legend()
    plt.ylabel("Likelihoods")
    plt.xlabel("Iterations")
    plt.show()
    # part c)
    print("Final accuracy on the validation data is:" + str(
        evaluate(val_data_unknown,
                 theta, alpha, beta)))
    print("Final accuracy on the training data is:" + str(
        evaluate(test_data_unknown,
                 theta, alpha, beta)))

if __name__ == "__main__":
    main()
