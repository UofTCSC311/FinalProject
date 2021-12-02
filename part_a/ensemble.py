from utils import *
import item_response


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
    """ return a list of predictions made by IRT
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: list
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = item_response.sigmoid(x)
        pred.append(p_a)
    return pred


def irt_predict(train_data, val_data, test_data, lr, num_iterations):
    """ return a tuple of validation, and test predictions
    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param test_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param num_iterations: int
    :return: (list, list)
    """
    theta, beta, irt_train_likelihoods, irt_valid_likelihoods, \
    irt_train_acc_lst1, irt_valid_acc_lst1 = \
        item_response.irt(train_data, val_data, lr, num_iterations)
    val_pred = irt_predictions(val_data, theta, beta)
    test_pred = irt_predictions(test_data, theta, beta)
    return val_pred, test_pred


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Bootstrap
    new_train_data1, new_train_data2, new_train_data3 = \
        bootstrap(train_data, len(train_data["question_id"])), \
        bootstrap(train_data, len(train_data["question_id"])),\
        bootstrap(train_data, len(train_data["question_id"]))

    lr = 0.01
    iterations = 25
    # Model 1: Item Response
    pred_model1 = irt_predict(new_train_data1, val_data, test_data, lr, iterations)

    # Model 2: Item Response
    pred_model2 = irt_predict(new_train_data2, val_data, test_data, lr, iterations)

    # Model 3: Item Response
    pred_model3 = irt_predict(new_train_data3, val_data, test_data, lr, iterations)

    # Base model (for comparison)
    pred_base_model = irt_predict(train_data, val_data, test_data, lr, iterations)

    # Bagged predictions
    bagged_val_pred = []
    for i in range(len(val_data["is_correct"])):
        bagged_val_pred.append(
            (pred_model1[0][i]+pred_model2[0][i]+pred_model3[0][i]) / 3 >= 0.5)

    val_score = evaluate(val_data, bagged_val_pred)
    print("Validation accuracy: {}".format(val_score))

    bagged_test_pred = []
    for i in range(len(test_data["is_correct"])):
        bagged_test_pred.append(
            (pred_model1[1][i] + pred_model2[1][i] + pred_model3[1][i]) / 3 >= 0.5)

    test_score = evaluate(test_data, bagged_test_pred)
    base_test_score = evaluate(test_data, pred_base_model[1])
    print("Bagging test accuracy: {}".format(test_score))
    print("Base model accuracy: {}".format(base_test_score))
    if test_score > base_test_score:
        print("Bagging obtained better performance")
    elif test_score < base_test_score:
        print("The base model obtained better performance")
    else:
        print("Both methods performed equally")


if __name__ == "__main__":
    main()
    
