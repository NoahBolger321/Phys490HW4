from data import MnistData
import pathlib
import json
import numpy as np
import torch
import torch.optim as optim
from var_auto_enc import VAE
import argparse
import torch.nn.functional as func
from torchvision.utils import save_image
import matplotlib.pyplot as plt

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
    parser.add_argument('-o', dest='output',
                        help='output directory', default="output/")
    parser.add_argument('-n', dest='num_samples', default=10,
                        help='number fo sample images produced')
    args = parser.parse_args()

    with open(args.param) as paramfile:
        param = json.load(paramfile)

    model = VAE(14*14)
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    def loss_func(recon_x, x, mu, log_var):
        BCE = func.binary_cross_entropy(recon_x.view(recon_x.size(0), 196), x.view(x.size(0), 196))
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    train_loss = []

    for epoch in range(1, int(param['num_epochs']) + 1):
            targets = torch.from_numpy(Y_train).long()

            recon_batch, mu, log_var = model.forward(X_train)

            loss = loss_func(recon_batch, X_train, mu, log_var)
            train_loss.append(loss)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}]'.format(epoch, param['num_epochs'])+\
                  '\tTraining Loss: {:.4f}'.format(loss.item()))

    pathlib.Path('output/').mkdir(parents=True, exist_ok=True)
    for i in range(1, int(args.num_samples) + 1):
        save_image(recon_batch.view(recon_batch.size(0), 1, 14, 14)[i], f'{args.output}{i}.png')

    plt.plot(range(1, int(param['num_epochs']) + 1), train_loss)
    plt.savefig(f'{args.output}loss.png')

    #TODO: save loss vs. epochs graph
    #TODO: take number from command line for sample images
    #TODO: figure out sample images
    #TODO: output dir (create if not exists)
