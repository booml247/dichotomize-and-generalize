# import packages
import argparse
from pbgdeep.dataset_loader import DatasetLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import numpy as np
import torch.utils.data
from torchvision.utils import save_image
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import time
import os
import errno
from evaluation import model_eval_SVR_StoNet
from tools import *
# from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import pandas as pd
from model import Net3
import pickle
from pbgdeep.dataset_loader import DatasetLoader
# from cg_batch import CG
# from thundersvm import SVR

# Basic Setting
parser = argparse.ArgumentParser(description='Simulation toy stonet')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--dataset', default='ads', type=str, help='Name of the dataset to use.')
parser.add_argument('--data_path', default="/scratch/gilbreth/liang257/SVR-StoNet/data/Mice/", type=str,
                    help='folder name for loading data')
parser.add_argument('--base_path', default='./results/',
                    type=str, help='base path for saving result')
parser.add_argument('--model_path', default='stonet_test/', type=str, help='folder name for saving model')
parser.add_argument('--data_index', default=0, type=int)

# Training Setting
parser.add_argument('--nepoch', default=101, type=int, help='total number of training epochs')
parser.add_argument('--subsample', default=10000, type=int, help='total number of training epochs')
parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.5, type=float, help='momentum in SGD')
parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay in SGD')
parser.add_argument('--batch_train', default=100, type=int, help='batch size for training')
parser.add_argument('--MH_step', default=25, type=int, help='SGLD step for imputation')
parser.add_argument('-C', default=1, type=float, help='value of hyperparameter C for SVR')
parser.add_argument('--epsilon', default=0.01, type=float, help='value of hyperparameter epsilon for SVR')

args = parser.parse_args([])


def main():
    import warnings
    import pickle
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Hyperparameter settings
    data_index = args.data_index
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # step_lr = args.lr / args.batch_train / args.MH_step
    step_lr = args.lr / args.batch_train
    num_epochs = args.nepoch
    subsample = args.subsample
    dataset = args.dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading dataset
    dataset_loader = DatasetLoader(random_state=args.seed)
    x_train, x_test, y_train, y_test = dataset_loader.load(dataset)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.2,
                                                          random_state=args.seed)


    x_train = torch.FloatTensor(x_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_train = torch.FloatTensor(y_train).to(device)

    x_test = torch.FloatTensor(x_test).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    y_test = torch.FloatTensor(y_test).to(device)


    # set the model path
    PATH = args.base_path + dataset + '/' + args.model_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise



    ntrain = x_train.shape[0]
    nval = x_val.shape[0]
    ntest = x_test.shape[0]
    dim = x_train.shape[1]


    # instantiate the net
    net = Net3(dim, x_train.shape[0]).to(device)
    # PATH = "/scratch/gilbreth/liang257/SVR-StoNet/MNIST_3_hidden_layer_temp1_50_10_50_10000data_gpu/0/stonet_test_MNIST/model59.pt"
    # net.load_state_dict(torch.load(PATH))

    # define the loss function
    loss_func = net.bound

    # set optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=step_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001 / args.batch_train)

    train_loss_path = np.zeros(num_epochs)
    train_acc_path = np.zeros(num_epochs)
    val_loss_path = np.zeros(num_epochs)
    test_loss_path = np.zeros(num_epochs)
    test_acc_path = np.zeros(num_epochs)
    best_accuracy = 0
    best_valid_accuracy = 0


    sse = nn.MSELoss(reduction='sum')


    proposal_lr = [0.000005, 0.0000005, 0.00000005]
    sigma_list = [0.00001, 0.000001, 0.0000001]
    temperature = [1, 1, 1]

    index = np.arange(ntrain)
    subn = ntrain
    subsample = index[0: subn]
    C = args.C
    epsilon = args.epsilon

    # construct SVR layer
    with torch.no_grad():
        svr_list = []
        for i in range(net.fc1.weight.shape[0]):
            temp = SVR(C=C, epsilon=epsilon, gamma="auto")
            temp.fit(x_train.cpu(), x_train.matmul(net.fc1.weight[i, :]).cpu())
            svr_list.append(temp)

    svr_out_train = np.zeros([subn, net.fc1.weight.shape[0]])
    svr_out_val = np.zeros([nval, net.fc1.weight.shape[0]])
    svr_out_test = np.zeros([ntest, net.fc1.weight.shape[0]])

    time_used_path = np.zeros(num_epochs)
    starttime = time.clock()
    m0 = x_train.shape[0]
    best_corr = 0
    for epoch in range(num_epochs):

        start_time = time.clock()

        for iter_index in range(ntrain // subn):

            if epoch == num_epochs - 1:
                torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '_' + str(iter_index) + '.pt')



            if epoch == 0:
                for i in range(net.fc1.weight.shape[0]):
                    svr_out_train[:, i] = svr_list[i].predict(x_train.cpu())

            hidden_list = []
            momentum_list = []
            with torch.no_grad():
                hidden_list.append(torch.FloatTensor(svr_out_train).to(device))
                hidden_list.append(net.fc2(torch.tanh(hidden_list[0])))
                hidden_list.append(net.fc3(torch.tanh(hidden_list[1])))
                momentum_list.append(torch.zeros_like(hidden_list[0]))

            hidden_1_debug = hidden_list[0].clone().cpu().data.numpy()
            hidden_2_debug = hidden_list[1].clone().cpu().data.numpy()
            hidden_3_debug = hidden_list[2].clone().cpu().data.numpy()

            for i in range(hidden_list.__len__()):
                hidden_list[i].requires_grad = True

            MH_step = args.MH_step

            alpha = 0.1

            forward_hidden = torch.FloatTensor(svr_out_train).to(device)
            # with torch.no_grad():
            #     print("before imputation:", sse(net.fc2(torch.tanh(hidden_list[0])), y_train[subsample,]))

            for repeat in range(MH_step):
                # print(hidden_list[0])
                net.zero_grad()

                # for repeat in range(MH_step):
                if hidden_list[2].grad is not None:
                    hidden_list[2].grad.zero_()

                hidden_likelihood = -sse(net.fc4(torch.tanh(hidden_list[2])), y_train) / sigma_list[
                    2] - sse(net.fc3(torch.tanh(hidden_list[1])), hidden_list[2]) / sigma_list[1]

                # hidden_likelihood = -sse(net.fc2(torch.tanh(hidden_list[0])), y_train[subsample,]) / sigma_list[-1] / 0.1

                # - sse(net.fc1(x_train[subsample, ]), hidden_list[0]) / sigma_list[0]
                hidden_likelihood.backward()

                step_proposal_lr = proposal_lr[2]
                with torch.no_grad():
                    hidden_list[2].data += step_proposal_lr / 2 * hidden_list[2].grad \
                                           + torch.FloatTensor(hidden_list[2].shape).to(device).normal_().mul(
                        np.sqrt(step_proposal_lr * temperature[2]))

                if hidden_list[1].grad is not None:
                    hidden_list[1].grad.zero_()
                hidden_likelihood = -sse(net.fc3(torch.tanh(hidden_list[1])), hidden_list[2]) / sigma_list[1] - sse(
                    net.fc2(torch.tanh(hidden_list[0])), hidden_list[1]) / sigma_list[0]
                hidden_likelihood.backward()

                step_proposal_lr = proposal_lr[1]
                with torch.no_grad():
                    hidden_list[1].data += step_proposal_lr / 2 * hidden_list[1].grad \
                                           + torch.FloatTensor(hidden_list[1].shape).to(device).normal_().mul(
                        np.sqrt(step_proposal_lr * temperature[1]))

                if hidden_list[0].grad is not None:
                    hidden_list[0].grad.zero_()
                hidden_likelihood = -sse(net.fc2(torch.tanh(hidden_list[0])), hidden_list[1]) / sigma_list[0]
                hidden_likelihood.backward()

                hidden_list[0].grad = hidden_list[0].grad + (-C) * torch.where(
                    hidden_list[0] - forward_hidden > epsilon,
                    torch.ones_like(forward_hidden),
                    torch.zeros_like(forward_hidden)) + (C) * torch.where(hidden_list[0] - forward_hidden < -epsilon,
                                                                          torch.ones_like(forward_hidden),
                                                                          torch.zeros_like(forward_hidden))

                step_proposal_lr = proposal_lr[0]
                with torch.no_grad():

                    momentum_list[0] = (1 - alpha) * momentum_list[0] + step_proposal_lr / 2 * hidden_list[
                        0].grad + torch.FloatTensor(
                        hidden_list[0].shape).to(device).normal_().mul(
                        np.sqrt(alpha * step_proposal_lr * temperature[0]))
                    hidden_list[0].data += momentum_list[0]

                    # hidden_list[0].data += step_proposal_lr / 2 * hidden_list[0].grad + torch.FloatTensor(
                    #     hidden_list[0].shape).to(device).normal_().mul(np.sqrt(step_proposal_lr * temperature[0]))
                # net_weight2_grad_imputation[iter_index,] = net.fc2.weight.grad.cpu().data.numpy()
                # print('hidden 0 grad max:', hidden_list[0].grad.abs().max(), 'hidden 0 grad std:', hidden_list[0].grad.std())

            print((torch.FloatTensor(hidden_1_debug).to(device) - hidden_list[0]).abs().max())
            print((torch.FloatTensor(hidden_2_debug).to(device) - hidden_list[1]).abs().max())
            print((torch.FloatTensor(hidden_3_debug).to(device) - hidden_list[2]).abs().max())

            with torch.no_grad():
                for i in range(net.fc1.weight.shape[0]):
                    svr_list[i].fit(x_train.cpu(), hidden_list[0][:, i].cpu().detach())

                print("before solving:", sse(net.fc2(torch.tanh(hidden_list[0])), hidden_list[1]))
                clf = Lasso(alpha=0.0001).fit(
                    torch.tanh(hidden_list[0]).detach().cpu(), hidden_list[1].data.detach().cpu())
                net.fc2.weight.data = torch.tensor(clf.coef_).float().reshape(net.fc2.weight.shape).to(device)
                net.fc2.bias.data = torch.tensor(clf.intercept_).float().to(device)
                print("after solving", sse(net.fc2(torch.tanh(hidden_list[0])), hidden_list[1]))

                print("before solving:", sse(net.fc3(torch.tanh(hidden_list[1])), hidden_list[2]))
                clf = Lasso(alpha=0.001).fit(
                    torch.tanh(hidden_list[1]).detach().cpu(), hidden_list[2].data.detach().cpu())
                net.fc3.weight.data = torch.FloatTensor(clf.coef_)
                net.fc3.bias.data = torch.FloatTensor(clf.intercept_)
                print("after solving:",
                      sse(net.fc3(torch.tanh(hidden_list[1]).detach().cpu()), hidden_list[2].detach().cpu()))

                print("before solving:", sse(net.fc4(torch.tanh(hidden_list[2])), y_train))
                clf = Lasso(alpha=0.01).fit(
                    torch.tanh(hidden_list[2]).detach().cpu(), y_train.detach().cpu())
                net.fc4.weight.data = torch.tensor(clf.coef_).float().reshape(net.fc4.weight.shape).to(device)
                net.fc4.bias.data = torch.tensor(clf.intercept_).float().float().reshape(net.fc4.bias.shape).to(device)
                print("after solving:", sse(net.fc4(torch.tanh(hidden_list[2])), y_train))

        with torch.no_grad():
            subsample = index[0: subn]
            print('epoch: ', epoch)
            for i in range(net.fc1.weight.shape[0]):
                svr_out_train[:, i] = svr_list[i].predict(x_train.cpu())

            net.to(device)
            output = net(torch.FloatTensor(svr_out_train).to(device))
            train_loss = loss_func(output, y_train)
            train_acc = (output == y_train).view(-1).sum().item()/len(y_train)

            filename = PATH + 'model_output_train' + str(epoch) + '.pt'
            f = open(filename, 'wb')
            pickle.dump(output, f, protocol=4)
            f.close()

            filename = PATH + 'model_subsample_train' + str(epoch) + '.pt'
            f = open(filename, 'wb')
            pickle.dump(subsample, f, protocol=4)
            f.close()

            # train_loss = model_eval_SVR_StoNet(svr_list, net, x_train, y_train)
            train_loss_path[epoch] = train_loss
            train_acc_path[epoch] = train_acc
            # print(output)
            print("train loss: ", train_loss)
            print("train accuracy", train_acc)



            for i in range(net.fc1.weight.shape[0]):
                svr_out_test[:, i] = svr_list[i].predict(x_test.cpu())

            output = net(torch.FloatTensor(svr_out_test).to(device))
            test_loss = loss_func(output, y_test)
            test_acc = (output == y_test).view(-1).sum().item() / len(y_test)

            filename = PATH + 'model_output_test' + str(epoch) + '.pt'
            f = open(filename, 'wb')
            pickle.dump(output, f, protocol=4)
            f.close()

            # test_loss = model_eval_SVR_StoNet(svr_list, net, x_test, y_test)
            test_loss_path[epoch] = test_loss
            test_acc_path[epoch] = test_acc
            print("test loss: ", test_loss)
            print("test accuracy", test_acc)


            # if epoch % 10 == 0:
            #     import pickle
            #     filename = PATH + 'model_svr.pt'
            #     f = open(filename, 'wb')
            #     pickle.dump(svr_list, f, protocol=4)
            #     f.close()

            # val_corr, val_loss = model_eval_SVR_StoNet(svr_list, net, x_val, y_val)
            # if val_corr > best_corr:
            #     best_corr = val_corr
            #     filename = PATH + 'best_val_model_svr.pt'
            #     f = open(filename, 'wb')
            #     pickle.dump(svr_list, f)
            #     f.close()
            #     torch.save(net.state_dict(),
            #                PATH + 'best_val_model_cv.pt')
            # print("val loss: ", val_loss, "; val correlation: ", val_corr)

        end_time = time.clock()

        time_used_path[epoch] = end_time - start_time

        torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

        import pickle
        filename = PATH + 'result_temp.txt'
        f = open(filename, 'wb')
        pickle.dump(
            [train_loss_path, test_loss_path, train_acc_path, test_acc_path, time_used_path], f)
        f.close()

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump(
        [train_loss_path, test_loss_path, time_used_path], f)
    f.close()


if __name__ == '__main__':
    main()






