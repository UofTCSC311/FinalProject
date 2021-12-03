from scipy.sparse import load_npz

import numpy as np
import csv
import os


def age_classifier(year):
    # Initialize the data.
    elder_list = []
    younger_list = []
    unknown_list = []
    with open("../data/student_meta.csv", "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                if int(row[2].split("-")[0]) <= year:
                    elder_list.append(row[0])
                elif int(row[2].split("-")[0]) > year:
                    younger_list.append(row[0])
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
        return elder_list, younger_list




def _load_csv(path):
    elder, younger = age_classifier(2013)
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    younger_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    elder_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    unknown_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                if row[1] in elder:
                    elder_data["question_id"].append(int(row[0]))
                    elder_data["user_id"].append(int(row[1]))
                    elder_data["is_correct"].append(int(row[2]))
                elif row[1] in younger:
                    younger_data["question_id"].append(int(row[0]))
                    younger_data["user_id"].append(int(row[1]))
                    younger_data["is_correct"].append(int(row[2]))
                else:
                    unknown_data["question_id"].append(int(row[0]))
                    unknown_data["user_id"].append(int(row[1]))
                    unknown_data["is_correct"].append(int(row[2]))
            except ValueError:
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return elder_data, younger_data, unknown_data


def load_train_sparse(root_dir="/data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str
    :return: 2D sparse matrix
    """
    path = os.path.join(root_dir, "train_sparse.npz")
    if not os.path.exists(path):
        raise Exception("The specified path {} "
                        "does not exist.".format(os.path.abspath(path)))
    matrix = load_npz(path)
    return matrix


def load_train_csv(root_dir="/data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "train_data.csv")
    return _load_csv(path)


def load_valid_csv(root_dir="/data"):
    """ Load the validation data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "valid_data.csv")
    return _load_csv(path)


def load_public_test_csv(root_dir="/data"):
    """ Load the test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "test_data.csv")
    return _load_csv(path)


def load_private_test_csv(root_dir="/data"):
    """ Load the private test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: an empty list.
    """
    path = os.path.join(root_dir, "private_test_data.csv")
    return _load_csv(path)


def save_private_test_csv(data, file_name="private_test_result.csv"):
    """ Save the private test data as a csv file.

    This should be your submission file to Kaggle.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    :param file_name: str
    :return: None
    """
    if not isinstance(data, dict):
        raise Exception("Data must be a dictionary.")
    cur_id = 1
    valid_id = ["0", "1"]
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "is_correct"])
        for i in range(len(data["user_id"])):
            if str(int(data["is_correct"][i])) not in valid_id:
                raise Exception("Your data['is_correct'] is not in a valid format.")
            writer.writerow([str(cur_id), str(int(data["is_correct"][i]))])
            cur_id += 1
    return


def evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param predictions: list
    :param threshold: float
    :return: float
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= threshold) == data["is_correct"])
            / float(len(data["is_correct"])))


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def sparse_matrix_predictions(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the predictions.

    This function can be used for submitting Kaggle competition.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: list
    """
    predictions = []
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    return predictions
