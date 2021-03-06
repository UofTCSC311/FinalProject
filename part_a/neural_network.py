import matplotlib.pyplot as plt

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encoded = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(encoded))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    valid_acc_lst = []
    train_loss_lst = []
    epoch_lst = []
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            loss = torch.sum((output - target) ** 2.) + lamb * 0.5 * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_acc_lst.append(valid_acc)
        epoch_lst.append(epoch)
        train_loss_lst.append(train_loss)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return valid_acc_lst, train_loss_lst, epoch_lst
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # # part c)
    # # Set model hyperparameters.
    k_lst = [10, 50, 100, 200, 500]
    # # Set optimization hyperparameters.
    lr_lst = [0.1, 0.5, 1]
    num_epoch_lst = [10, 20, 50]
    lamb = 0
    # we didn't use regularizer for this part
    # plot and find model with the best accuracy
    part1, summary = plt.subplots(len(num_epoch_lst), len(lr_lst))
    for id_epoch in range(len(num_epoch_lst)):
        epoch = num_epoch_lst[id_epoch]
        for id_lr in range(len(lr_lst)):
            lr = lr_lst[id_lr]
            for k in k_lst:
                cols = train_matrix.shape[1]
                model = AutoEncoder(cols, k)
                acc_valid, training_loss, epoch_lst = train(model, lr, lamb, train_matrix, zero_train_matrix,valid_data, epoch)
                summary[id_epoch][id_lr].plot(epoch_lst, acc_valid)
                summary[id_epoch][id_lr].set_ylabel("Accuracy")
                summary[id_epoch][id_lr].set_title(f"Epoch: {epoch}, Lr: {lr}")
            summary[id_epoch][id_lr].legend(loc="upper left")
    summary.show()
    # from the plot, we know  Epoch = 20, lr = 0.1, need to find k*
    epoch, lr = 20, 0.1
    for k in k_lst:
        cols = train_matrix.shape[1]
        model = AutoEncoder(cols, k)
        acc_valid, training_loss, epoch_lst = train(model, lr, lamb,
                                                    train_matrix,
                                                    zero_train_matrix,
                                                    valid_data, epoch)
        plt.plot(epoch_lst, acc_valid, label=f"k value:{k}")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.title(
            f"Epoch value : {epoch}, Lr value: {lr}, Lambda Value: 0")

    plt.savefig('Q3 c')
    # from plot, the best accuracy is around 0.684, corresponding hyperparameter
    # Epoch = 20, lr = 0.1, k* = 10

    # # part d)
    k = 10
    cols = train_matrix.shape[1]
    model = AutoEncoder(cols, k)
    fig, pltd = plt.subplots(ncols=2, figsize=(20, 15))
    acc_valid, training_loss, epoch_lst = train(model, lr, lamb,
                                                train_matrix,
                                                zero_train_matrix,
                                                valid_data, epoch)
    print(f"Highest validation accuracy:{max(acc_valid)}")
    pltd[0].plot(epoch_lst, acc_valid, label=f"k value:{k}")
    pltd[0].set_ylabel("Accuracy")
    pltd[0].set_xlabel("Epoch")

    pltd[1].plot(epoch_lst, training_loss, label=f"k value:{k}")
    pltd[1].set_ylabel("Training loss")
    pltd[1].set_xlabel("Epoch")
    fig.savefig('Q3 d')
    print(f"Final test accuracy: {evaluate(model, zero_train_matrix, test_data)}")
    # part e)
    lamb_lst = [0.001, 0.01, 0.1, 1]
    for lamb in lamb_lst:
        k = 10
        cols = train_matrix.shape[1]
        model = AutoEncoder(cols, k)
        acc_valid, training_loss, epoch_lst = train(model, lr, lamb,
                                                    train_matrix,
                                                    zero_train_matrix,
                                                    valid_data, epoch)
        print(f"The best accuracy: {max(acc_valid)} for lambda = {lamb}")
        plt.plot(epoch_lst, acc_valid, label=f"k value:{k}")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.title(f"Epoch value : {epoch}, Lr value: {lr}, Lambda value: {lamb}")
        plt.show()
        # report final accuracy
    print(f"Final test accuracy with regularizer: {evaluate(model, zero_train_matrix, test_data)}")
        # from comparison, lambda = 0.001 is good
    k = 10
    model = AutoEncoder(train_matrix.shape[1], k)
    acc_valid, training_loss, epoch_lst = train(model, lr, lamb,
                                                train_matrix,
                                                zero_train_matrix,
                                                valid_data, epoch)
    plt.plot(epoch_lst, acc_valid, label=f"k value:{k}")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title(f"Epoch value : {epoch}, Lr value: {lr}, Lambda value: {lamb}")
    plt.show()
    plt.plot(epoch_lst, training_loss, label=f"k value:{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(f"Epoch value : {epoch}, Lr value: {lr}")

    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
