# TODO: complete this file.
import random
from utils import *
import item_response
import neural_network
import knn
import matrix_factorization


def bootstrap_matrix(matrix):
    """ boostrap samples for a sparse matrix
    :param matrix: 2D sparse matrix
    :return: 2D sparse matrix
    """
    # TODO
    pass


def bootstrap(data):
    """ boostrap samples from a dataset
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    # TODO
    length = len(data["question_id"])
    new_data = {"user_id": [], "question_id": [], "is_correct": []}
    for i in range(length):
        i = np.random.choice(length, 1)[0]
        new_data["user_id"].append(data["user_id"][i])
        new_data["question_id"].append(data["question_id"][i])
        new_data["is_correct"].append(data["is_correct"][i])
    return new_data


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    model1 = knn.knn_impute_by_user
    model2 = item_response.irt
    model3 = neural_network.train

    # temporary set seed
    random.seed(1)

    train_data1, train_data2, train_data3 = bootstrap(train_data), \
                                            bootstrap(train_data), \
                                            bootstrap(train_data)

    irt_lr = 0.01
    irt_iterations = 25
    irt_theta, irt_beta, irt_train_likelihoods, irt_valid_likelihoods, \
    irt_train_acc_lst, irt_valid_acc_lst = item_response.irt(
        train_data, val_data, irt_lr, irt_iterations)
    print("Final accuracy on the validation data is:" +
          str(item_response.evaluate(val_data, irt_theta, irt_beta)))
    print("Final accuracy on the training data is:" +
          str(item_response.evaluate(test_data, irt_theta, irt_beta)))

    pred1 = None
    pred2 = None
    pred3 = None


if __name__ == "__main__":
    main()
