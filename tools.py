import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def compute_distance(X, Y):
    n_1, n_2 = X.size(0), Y.size(0)

    norms_1 = torch.sum(X ** 2, dim=1, keepdim=True)
    norms_2 = torch.sum(Y ** 2, dim=1, keepdim=True)
    norms = (norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2))
    distances_squared = norms - 2 * X @ Y.t()

    return torch.abs(distances_squared)

def RBF_kernel(x, y , gamma):
    distance = compute_distance(x, y)
    kernel = torch.exp(-gamma * distance)
    return kernel


def plot_loss(load_path, save_path):
    # load loss paths
    f = open(load_path, 'rb')
    import pickle
    train_loss_path, test_loss_path, time = pickle.load(f)

    plt.figure(0)
    plt.plot(train_loss_path[:40], label="train_loss")
    plt.plot(test_loss_path[:40], label="test_loss")
    plt.legend(loc="upper right")
    plt.xlabel('Number of epochs')
    plt.ylabel('MSE')
    plt.show()
    plt.savefig(save_path)
    plt.close()
