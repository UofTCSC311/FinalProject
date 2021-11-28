from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    grad = c - np.sum(u[n] * z[q])
    u_temp = u.copy()
    u += lr * grad * z[q]
    z += lr * grad * u_temp[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, plot=False, val_data=None):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param plot: bool
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list} only used for computing square loss
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_plot_lst = []
    val_plot_lst = []
    for _ in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if plot:  # compute square losses
            train_plot_lst.append(squared_error_loss(train_data, u, z))
            val_plot_lst.append(squared_error_loss(val_data, u, z))
    if plot:  # plot square losses
        plt.title("Squared error losses as a function of iteration")
        plt.xlabel("Iterations")
        plt.ylabel("Square error loss")
        iterations = [i for i in range(num_iteration)]
        plt.plot(iterations, train_plot_lst, label="Training error")
        plt.plot(iterations, val_plot_lst, label="Validation error")
        plt.legend()
        plt.show()
    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_values = [i for i in range(1, 10)]
    svd_validation = []
    for k in k_values:
        svd_matrix = svd_reconstruct(train_matrix, k)
        svd_validation.append(sparse_matrix_evaluate(val_data, svd_matrix))

    best_svd_recon_k = np.argmax(svd_validation)
    best_svd_recon_acc = svd_validation[best_svd_recon_k]
    svd_recon_test = sparse_matrix_evaluate(
        test_data, svd_reconstruct(train_matrix, k_values[best_svd_recon_k]))

    print("Best svd reconstruct k: {}".format(k_values[best_svd_recon_k]))
    print("Best svd reconstruct accuracy: {}".format(best_svd_recon_acc))
    print("Best svd reconstruct k on test: {}".format(svd_recon_test))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    new_k_values = [i for i in range(1, 50)]
    lr, num_iterations = 0.01, 1000
    als_validation = []
    for k in new_k_values:
        als_matrix = als(train_data, k, lr, num_iterations)
        als_validation.append(sparse_matrix_evaluate(val_data, als_matrix))

    best_als_k = np.argmax(als_validation)
    best_als_acc = als_validation[best_als_k]
    test_acc = sparse_matrix_evaluate(test_data, als(train_data, best_als_k, lr, num_iterations))

    print("Learning rate: {}".format(lr))
    print("Iterations: {}".format(num_iterations))
    print("Best als k: {}".format(new_k_values[best_als_k]))
    print("Best als accuracy: {}".format(best_als_acc))
    print("Test accuracy: {}".format(test_acc))
    # call als function to plot train and validation square errors
    als(train_data, best_als_k, lr, num_iterations, plot=True, val_data=val_data)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
    
