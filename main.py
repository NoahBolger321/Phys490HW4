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

    # pull data and convert to proper objects
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

    # initialize model and optimizer
    model = VAE(14*14)
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    def loss_func(recon_x, x, mu, log_var):
        BCE = func.binary_cross_entropy(recon_x.view(recon_x.size(0), 196), x.view(x.size(0), 196))
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

    # storage lists for loss and batches
    train_loss = []
    test_loss = []
    batches = []

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=param['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=X_test, batch_size=param['batch_size'], shuffle=False)

    # perform training
    for epoch in range(1, int(param['num_epochs']) + 1):
        for batch_idx, data in enumerate(train_loader):

            # retrive output from model
            recon_data, mu, log_var = model.forward(data)

            #calculate loss
            loss = loss_func(recon_data, data, mu, log_var)
            train_loss.append(loss)
            optimizer.zero_grad()
            batches.append(batch_idx)

            loss.backward()
            optimizer.step()

            # verbose output for reporting loss at specific batch indices and through epochs
            if batch_idx % 100 == 0:
                print('Epoch [{}/{} ({}%)]'.format(epoch, param['num_epochs'],
                                                   round(100*(batch_idx*len(data))/len(train_loader.dataset)))+\
                  '\tTraining Loss: {:.4f}'.format(loss.item()))

    # create /output if not exists
    pathlib.Path('output/').mkdir(parents=True, exist_ok=True)

    # testing
    with torch.no_grad():
        for data in test_loader:
            test_recon_data, mu, log_var = model.forward(data)
            loss = loss_func(test_recon_data, data, mu, log_var)
            test_loss.append(loss)

        # return separate images (number from command line arg) as png's and save in output dir
        i = int(args.num_samples)
        for img in test_recon_data.view(test_recon_data.size(0), 1, 14, 14)[:int(args.num_samples)]:
            save_image(img, f'{args.output}{i}.png')
            i = i - 1
        print("Test Loss: {}".format(float(test_loss[len(test_loss)-1])))

    plt.plot(batches, train_loss)
    plt.xlabel('Batches')
    plt.ylabel('Training Loss')
    plt.savefig(f'{args.output}loss.png')
