import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_size, intermediate_size, encoding_size):
        self.fc1 = nn.Linear(input_size, intermediate_size)
        self.fc21 = nn.Linear(intermediate_size, encoding_size)
        self.fc22 = nn.Linear(intermediate_size, encoding_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, encoding_size, intermediate_size, output_size):
        self.fc1 = nn.Linear(encoding_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, x):
        h3 = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h3))


class VAE:
    INTERMEDIATE_SIZE = 15
    ENCODING_SIZE = 8

    def __init__(self, input_size,
                 intermediate_size=None,
                 encoding_size=None, alpha=0.1):

        if not intermediate_size:
            intermediate_size = self.INTERMEDIATE_SIZE
        if not encoding_size:
            encoding_size = self.ENCODING_SIZE

        self.encoder = Encoder(input_size, intermediate_size, encoding_size)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=alpha)

        self.decoder = Decoder(encoding_size, intermediate_size, input_size)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=alpha)

        self.input_size = input_size

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train(self, x, opt):
        opt.zero_grad()

        recon_batch, mu, logvar = self(x)
        loss = self.loss_function(recon_batch, x, mu, logvar)
        loss.backward()

        opt.step()

        return loss
