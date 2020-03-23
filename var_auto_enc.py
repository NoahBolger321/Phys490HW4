import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, n_pixels):
        super(VAE, self).__init__()
        self.n_pixels = n_pixels

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 196)

        self.encoder = nn.Sequential(
            # encoder layers
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # decoder layers
            nn.ConvTranspose2d(196, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 32, 4, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 2, 2, 0),
            nn.Sigmoid()
        )

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        encoder = self.encoder(x)
        x = encoder.view(encoder.size(0), -1)
        return self.fc1(x), self.fc2(x)

    def decode(self, x):
        x = self.fc3(x)
        decoder = self.decoder(x.view(x.size(0), 196, 1, 1))
        return decoder

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sampling(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

    # def test(self, X_test, Y_test, loss):
    #     self.eval()
    #     with torch.no_grad():
    #         targets = torch.from_numpy(Y_test).long()
    #         output = self.forward(X_test)
    #         cross_val = loss(output, targets)
    #     return cross_val.item(), output, Y_test
