import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(np.transpose(matrix)).transpose()
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # Part a)
    k_values = [1, 6, 11, 16, 21, 26]
    user_validation = []
    item_validation = []

    for k in k_values:
        user_validation.append(knn_impute_by_user(sparse_matrix, val_data, k))

    plt.title("Validation Classification rate against k on student similarity")
    plt.xlabel("k")
    plt.ylabel("Classification rate")
    plt.scatter(k_values, user_validation)
    plt.show()
    best_user_k = np.argmax(user_validation)
    best_user_acc = user_validation[int(best_user_k)]

    # Part b)
    print("Best user k: {}".format(k_values[best_user_k]))
    print('Best user accuracy: {}'.format(str(best_user_acc)))

    # Part c)
    for k in k_values:
        item_validation.append(knn_impute_by_item(sparse_matrix, val_data, k))

    plt.title("Validation Classification rate against k on question similarity")
    plt.xlabel("k")
    plt.ylabel("Classification rate")
    plt.scatter(k_values, item_validation)
    best_item_k = np.argmax(item_validation)
    best_item_acc = item_validation[int(best_item_k)]
    plt.show()

    print("Best item k: {}".format(k_values[best_item_k]))
    print('Best item accuracy: {}'.format(str(best_item_acc)))

    # Part d)
    test_user_acc = knn_impute_by_user(sparse_matrix, test_data, k_values[best_user_k])
    test_item_acc = knn_impute_by_item(sparse_matrix, test_data, k_values[best_item_k])
    if test_user_acc > test_item_acc:
        print("user based collaborative filter performs better")
    elif test_user_acc < test_item_acc:
        print("item based collaborative filter performs better")
    else:
        print("both methods perform equally")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
