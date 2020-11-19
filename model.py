import torch.nn as nn
import torch
from pbgdeep.utils import linear_loss, bound
from torch.nn.parameter import Parameter
import itertools


class Net(nn.Module):
    def __init__(self, block_list):
        super(Net, self).__init__()
        self.block_list = block_list
        for i in range(len(block_list)):
            self.add_module('block' + str(i), block_list[i])

    def forward(self, x):
        for i in range(len(self.block_list)):
            x = self.block_list[i](x)
        return x


class Net2(nn.Module):
    def __init__(self, dim):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(dim, 5)
        self.fc2 = nn.Linear(5,1)

    def forward(self, x):
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class Net3(nn.Module):
    def __init__(self, input_size, n_examples, hidden_size=[50,30,50], delta=0.05):
        super(Net3, self).__init__()
        self.delta = delta
        self.t = Parameter(torch.Tensor(1))
        self.n_examples = n_examples
        self.hidden_size = hidden_size


        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1],hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2],1)

    def forward(self, x):
        x = torch.sign(x)
        x = torch.sign(self.fc2(x))
        x = torch.sign(self.fc3(x))
        x = torch.sign(self.fc4(x))
        return x

    def bound(self, pred_y, y):
        """Bound computation as presented in Theorem 3. with the learned C value."""
        loss = linear_loss(pred_y, y)

        C = torch.exp(self.t)
        kl = self.compute_kl()
        bound_value = bound(loss, kl, self.delta, self.n_examples, C)
        return bound_value

    def compute_kl(self):
        """Kullback-Leibler divergence computation as presented in equation 17."""
        kl = 0
        coefficient = 1

        norm = torch.norm(self.fc1.weight - 0) ** 2 + torch.norm(self.fc1.bias - 0) ** 2
        kl += coefficient * norm
        coefficient *= self.hidden_size[0]

        norm = torch.norm(self.fc2.weight - 0) ** 2 + torch.norm(self.fc2.bias - 0) ** 2
        kl += coefficient * norm
        coefficient *= self.hidden_size[1]

        norm = torch.norm(self.fc3.weight - 0) ** 2 + torch.norm(self.fc3.bias - 0) ** 2
        kl += coefficient * norm
        coefficient *= self.hidden_size[2]

        norm = torch.norm(self.fc4.weight - 0) ** 2 + torch.norm(self.fc4.bias - 0) ** 2
        kl += coefficient * norm


        return 0.5 * kl

class stonet(nn.Module):
    def __init__(self, dim):
        super(stonet, self).__init__()
        self.fc1 = nn.Linear(dim, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        x = torch.tanh(x)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x



class DNN1(nn.Module):
    def __init__(self, dim):
        super(DNN1, self).__init__()
        self.fc1 = nn.Linear(dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class DNN2(nn.Module):
    def __init__(self, dim):
        super(DNN2, self).__init__()
        self.fc1 = nn.Linear(dim, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class DNN3(nn.Module):
    def __init__(self, dim):
        super(DNN3, self).__init__()
        self.fc1 = nn.Linear(dim, 50)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


def load_blocks(input_dim):
    block_list = []
    block_list.append(nn.Sequential(nn.Linear(10, 30), nn.Tanh()))
    block_list.append(nn.Sequential(nn.Linear(30, 50),nn.Tanh()))
    block_list.append(nn.Sequential(nn.Linear(50, input_dim)))

    # block_list.append(nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Conv2d(20, 50, 5, 1)))
    # block_list.append(nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)), Resize()))
    # block_list.append(nn.Sequential(nn.ReLU(), nn.MaxPool2d((2,2)), Resize(), nn.Linear(4*4*50, 500) ) )
    # block_list.append(nn.Sequential(nn.ReLU(), nn.Linear(500, 10)))
    return block_list