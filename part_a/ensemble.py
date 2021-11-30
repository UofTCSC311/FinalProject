# TODO: complete this file.
from utils import *
import item_response
import neural_network
import knn


def bootstrap_matrix(matrix, m):
    """ boostrap samples for a sparse matrix
    :param matrix: 2D sparse matrix
    :param m: int
    :return: 2D sparse matrix
    """
    new_matrix_lst = []
    for _ in range(m):
        i = np.random.choice(matrix.shape[0], 1)[0]
        new_matrix_lst.append(matrix[i])
    new_matrix = np.array(new_matrix_lst)
    return new_matrix


def bootstrap(data, m):
    """ boostrap samples from a dataset
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param m: int
    :return: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    new_data = {"user_id": [], "question_id": [], "is_correct": []}
    for _ in range(m):
        i = np.random.choice(m, 1)[0]
        new_data["user_id"].append(data["user_id"][i])
        new_data["question_id"].append(data["question_id"][i])
        new_data["is_correct"].append(data["is_correct"][i])
    return new_data


def irt_predictions(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = item_response.sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Bootstrap
    new_train_data1, new_train_data2 = \
        bootstrap(train_data, len(train_data["question_id"])), \
        bootstrap(train_data, len(train_data["question_id"]))

    new_train_matrix1 = bootstrap_matrix(train_matrix, train_matrix.shape[0])

    # Model 1: Item Response
    irt_lr = 0.01
    irt_iterations = 25
    irt_theta, irt_beta, irt_train_likelihoods, irt_valid_likelihoods, \
    irt_train_acc_lst, irt_valid_acc_lst = \
        item_response.irt(new_train_data1, val_data, irt_lr, irt_iterations)
    irt_val_pred = irt_predictions(val_data, irt_theta, irt_beta)
    irt_test_pred = irt_predictions(test_data, irt_theta, irt_beta)

    # Model 2: kNN
    nbrs = knn.KNNImputer(n_neighbors=5)  # temporary
    knn_mat = nbrs.fit_transform(new_train_matrix1)
    knn_val_pred = sparse_matrix_predictions(val_data, knn_mat)
    knn_test_pred = sparse_matrix_predictions(test_data, knn_mat)

    # Model 3:

    # Bagged predictions
    bagged_val_pred = []
    for i in range(len(val_data["is_correct"])):
        bagged_val_pred.append((irt_val_pred[i] + knn_val_pred[i]) / 2 >= 0.5)

    val_score = evaluate(val_data, bagged_val_pred)
    print("Validation accuracy: {}".format(val_score))

    bagged_test_pred = []
    for i in range(len(test_data["is_correct"])):
        bagged_test_pred.append((irt_test_pred[i] + knn_test_pred[i]) / 2 >= 0.5)

    test_score = evaluate(test_data, bagged_test_pred)
    print("Test accuracy: {}".format(test_score))


if __name__ == "__main__":
    main()
