import csv

import matplotlib.pyplot as plt

from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def sigmoid_new(x, gamma_i):
    return np.exp(x) / (gamma_i + np.exp(x))


def expression_helper_1(diff, gamma_i):
    return np.log((np.exp(diff) + gamma_i)/(1+np.exp(diff)))


def expression_helper_2(diff, gamma_i):
    return np.log((1-gamma_i)/(1+np.exp(diff)))


def expression_helper_3(diff, gamma_i):
    return np.log(np.exp(diff) /(gamma_i+np.exp(diff)))


def inverse(x):
    return np.ones(np.shape(x)[0]) / (x - 1)


def gamma_inverse(diff, gamma):
    return np.ones(np.shape(diff)[0]) / (gamma + np.exp(diff))


def neg_log_likelihood(data, theta, beta, gamma):
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
        diff = (theta[user_key] - beta[question_key])

        log_lklihood += correct * expression_helper_1(diff, gamma[user_key]) + (1-correct) * expression_helper_2(diff, gamma[user_key])
    return -log_lklihood


def get_mean(arr, weight):
    dic = {}
    lst = [1 for i in range(56688)]
    for i in range(len(arr)):
        dic[arr[i]] = []
    for i in range(len(arr)):
        dic[arr[i]].append(weight[i])
    for key, value in dic.items():
        try:
            dic[key] = sum(dic[key]) / len(dic[key])
        except ValueError:
            dic[key] = 0.0
    for i in range(56688):
        lst[i] = dic[arr[i]]
    return lst


def update_theta_beta_bound(data, lr, theta, beta, gamma):
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
    a = data["user_id"]
    theta_array = np.array(theta[data["user_id"]])
    beta_array = np.array(beta[data["question_id"]])
    gamma_i = np.array(gamma[data["user_id"]])
    diff = (theta_array - beta_array)
    likli = sigmoid(diff)
    gamma_likeli = sigmoid_new(diff, gamma_i)
    derivative_theta = data["is_correct"] * np.exp(expression_helper_3(diff, gamma_i)) - sigmoid(diff)
    derivative_beta = -derivative_theta
    derivative_gamma = data["is_correct"] * (gamma_inverse(diff, gamma_i) - inverse(gamma_i)) + inverse(gamma_i)
    gamma_list = derivative_gamma.tolist()
    theta += lr * np.bincount(data["user_id"], derivative_theta)
    beta += lr * np.bincount(data["question_id"], derivative_beta)
    derivative = np.array(get_mean(data["user_id"], derivative_gamma))

    gamma += lr * np.bincount(data["user_id"], derivative)
    for i in range(len(gamma)):
        if gamma[i] < 0:
            gamma[i] = 0
        elif gamma[i] > 0.9:
            gamma[i] = 0.9
    return theta, beta, gamma


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
    theta = np.zeros(rows)
    beta = np.zeros(columns)
    gamma = np.ones(rows) * 0.001
    train_likelihood = []
    valid_likelihood = []
    val_acc_lst = []
    train_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, gamma=gamma)
        train_likelihood.append(neg_lld)
        valid_lld = neg_log_likelihood(val_data, theta, beta, gamma)
        valid_likelihood.append(valid_lld)
        score_train = evaluate(data, theta, beta, gamma)
        train_acc_lst.append(score_train)
        score = evaluate(val_data, theta, beta, gamma)
        val_acc_lst.append(score)
        print("On training data set, NLLK: {} \t Score: {}".format(neg_lld,
                                                                   score))
        print(f"Mean of bottom line: {np.mean(gamma)}")
        print(f"Max of bottom line: {np.max(gamma)}")
        print(f"Min of bottom line: {np.min(gamma)}")
        theta,  beta, gamma = update_theta_beta_bound(data, lr, theta, beta, gamma)

    return theta,  beta, gamma, train_likelihood, valid_likelihood, train_acc_lst, val_acc_lst


def evaluate(data, theta, beta, gamma):
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
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x) * (1 - gamma[u]) + gamma[u]
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    # set hyperparameter
    lr, iterations = 0.0003, 25
    lst_iterations = [i for i in range(1, 26)]
    theta, beta, gamma, train_likelihoods, valid_likelihoods, train_acc_lst, \
        valid_acc_lst = irt(train_data, val_data, lr, iterations)
    plt.title("1-Parameter IRT")
    plt.plot(lst_iterations, train_likelihoods, label="training set likelihood")
    plt.plot(lst_iterations, valid_likelihoods, label="validation set likelihood")
    plt.legend()
    plt.ylabel("Likelihoods")
    plt.xlabel("Iterations")
    plt.show()
    # part c)
    print("Final accuracy on the validation data is:" + str(evaluate(val_data,
                                                                theta,beta, gamma)))
    print("Final accuracy on the training data is:" + str(evaluate(test_data,
                                                                theta,beta, gamma)))

if __name__ == "__main__":
    main()
