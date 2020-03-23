from data import MnistData
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from var_auto_enc import VAE
import argparse
# from helpers import calc_accuracy
import torch.nn.functional as func

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

    model = VAE(14*14)
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    def loss_func(recon_x, x, mu, log_var):
        BCE = func.binary_cross_entropy(recon_x.view(recon_x.size(0), 196), x.view(x.size(0), 196))
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    test_losses = []
    test_accuracys = []

    for epoch in range(1, int(param['num_epochs']) + 1):
            targets = torch.from_numpy(Y_train).long()

            recon_batch, mu, log_var = model.forward(X_train)

            loss = loss_func(recon_batch, X_train, mu, log_var)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}]'.format(epoch, param['num_epochs'])+\
                  '\tTraining Loss: {:.4f}'.format(loss.item()))
                  # '\tTest Loss: {:.4f}'.format(test_loss))
                  # '\tAccuracy: {:.4f}'.format(accuracy))
    print(recon_batch)
    # print('\tFinal test loss: {:.4f}'.format(test_losses[-1]) + \
    #       '\tFinal test accuracy: {:.4f}'.format(test_accuracys[-1]))
