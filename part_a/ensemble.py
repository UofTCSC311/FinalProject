# TODO: complete this file.
from utils import *
import item_response
import neural_network
import knn
import matrix_factorization


def boostrap_matrix(matrix, m):
    """ boostrap samples for a sparse matrix
    :param matrix: 2D sparse matrix
    :param m: int
    :return: 2D sparse matrix
    """
    # TODO
    pass


def boostrap(data, m):
    """ boostrap samples from a dataset
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param m: int
    :return: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    # TODO
    pass


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    model1 = knn.knn_impute_by_user
    model2 = item_response.irt
    model3 = neural_network.train

    train_data1 = np.random.sample()
    val_data1 = None
    test_data1 = None
    train_data2 = None
    val_data2 = None
    test_data2 = None
    train_data3 = None
    val_data3 = None
    test_data3 = None

    pred1 = None
    pred2 = None
    pred3 = None


if __name__ == "__main__":
    main()
