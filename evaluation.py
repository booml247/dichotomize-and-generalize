import pickle
import torch
from model import Net3
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np

def model_eval_DNN(DNN_model, x, y):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = DNN_model(torch.FloatTensor(x).to(device))


    return pearsonr(output.detach().numpy().reshape(-1,1), y.detach().numpy().reshape(-1,1))[0],mean_squared_error(y.detach().numpy(), output.detach().numpy())


def model_eval_SVR_StoNet(svr_model, DNN_model, x, y):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = x.shape[0]
    svr_out_test = np.zeros([n, len(svr_model)])
    for i in range(len(svr_model)):
        svr_out_test[:, i] = svr_model[i].predict(x)

    output = DNN_model(torch.FloatTensor(svr_out_test).to(device))
    # pearsonr(output.detach().numpy().squeeze(), y.detach().numpy().squeeze())[0],

    return mean_squared_error(y.detach().numpy(), output.detach().numpy())