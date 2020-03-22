from data import MnistData
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from var_auto_enc import VAE
import argparse
# from helpers import calc_accuracy


if __name__ == '__main__':

    data = MnistData(test_size=0.1017)
    # print(data.X_train.values))

    X_train = torch.tensor(list(data.X_train.values)).view(-1,1,14,14).float()
    Y_train = np.matrix(data.Y_train.tolist())
    X_test = torch.tensor(list(data.X_test.values)).view(-1,1,14,14).float()
    Y_test = np.matrix(data.Y_test.tolist())


    # Command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Tutorial')
    parser.add_argument('-p', dest='param', metavar='data/parameters.json',
                        help='parameter file name')
    args = parser.parse_args()

    with open(args.param) as paramfile:
        param = json.load(paramfile)

    # model = VAE(14*14)
    # optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    # loss_func = nn.CrossEntropyLoss()
    #
    # test_losses = []
    # test_accuracys = []
    #
    # for epoch in range(1, int(param['num_epochs']) + 1):
    #
    #         inputs = torch.from_numpy(X_train)
    #         targets = torch.from_numpy(Y_train).long()
    #
    #         output = model.forward(X_train)
    #
    #         loss = loss_func(output, targets.reshape(-1))
    #         optimizer.zero_grad()
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         if (epoch + 1) % 10 == 0:
    #
    #             test_loss, test_output, test_targets = model.test(X_test, Y_test, loss_func)
    #             test_losses.append(test_loss)
    #             # test_accuracys.append(calc_accuracy(test_output, test_targets))
    #             #
    #             # accuracy = calc_accuracy(output, targets)
    #             print('Epoch [{}/{}]'.format(epoch+1, param['num_epochs'])+\
    #                   '\tTraining Loss: {:.4f}'.format(loss.item())+\
    #                   '\tTest Loss: {:.4f}'.format(test_loss))
    #                   # '\tAccuracy: {:.4f}'.format(accuracy))
    #
    # print('\tFinal test loss: {:.4f}'.format(test_losses[-1]) + \
    #       '\tFinal test accuracy: {:.4f}'.format(test_accuracys[-1]))
