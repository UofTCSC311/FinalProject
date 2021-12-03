import matplotlib.pyplot as plt

from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data["user_id"])):
        correct = data["is_correct"][i]
        user_key = data["user_id"][i]
        question_key = data["question_id"][i]
        # from the equation derived in a)
        diff = (theta[user_key] - beta[question_key])
        log_lklihood += correct * diff - np.log(1 + np.exp(diff))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
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
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # from the equation, first derive sigmoid of thea -beta
    theta_array = np.array(theta[data["user_id"]])
    beta_array = np.array(beta[data["question_id"]])
    likli = sigmoid(theta_array - beta_array)
    derivative_theta = data["is_correct"] - likli
    derivative_beta = -derivative_theta
    # likeli is a large array needs grouping use bincount
    theta += lr * np.bincount(data["user_id"], derivative_theta)
    beta += lr * np.bincount(data["question_id"], derivative_beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


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
    # TODO: Initialize theta and beta.
    # first find dimension of data
    rows = max(data["user_id"]) + 1
    # rows = 542
    columns = max(data["question_id"]) + 1
    # columns = 1774
    # use random to set
    np.random.seed(1)
    theta = np.random.rand(rows)
    beta = np.random.rand(columns)
    train_likelihood = []
    valid_likelihood = []
    val_acc_lst = []
    train_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_likelihood.append(neg_lld)
        valid_lld = neg_log_likelihood(val_data, theta, beta)
        valid_likelihood.append(valid_lld)
        score_train = evaluate(data, theta, beta)
        train_acc_lst.append(score_train)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("On training data set, NLLK: {} \t Score: {}".format(neg_lld,
                                                                   score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_likelihood, valid_likelihood, train_acc_lst, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # set hyperparameter
    lr, iterations = 0.01, 25
    lst_iterations = [i for i in range(1, 26)]
    theta, beta, train_likelihoods, valid_likelihoods, train_acc_lst, \
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
                                                                theta,beta)))
    print("Final accuracy on the training data is:" + str(evaluate(test_data,
                                                                theta,beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    question_lst = [500, 1000, 1500]
    # reshape data structure of theta
    theta.reshape(-1).sort()
    plt.title("Probability curve of questions")
    for question in question_lst:
        plt.plot(theta, sigmoid(theta - beta[question]), label="Question" + str(question))
    plt.xlabel("Theta")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
    
