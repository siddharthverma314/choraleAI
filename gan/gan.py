import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import data_loader

DATA_FILE = './data/gan_data'


class GAN():

    BATCH_SIZE = 3

    def __init__(self, discriminator, generator):

        self.discriminator = discriminator
        self.generator = generator

        self._loss = F.cross_entropy

        self._data_label = 1
        self._gen_label = 0

        self._d_train = torch.optim.Adam(self.discriminator.parameters())
        self._g_train = torch.optim.Adam(self.generator.parameters())

    def train(self, data, batch_size):
        # create generated image
        self.discriminator.zero_grad()
        self.generator.zero_grad()

        # import ipdb; ipdb.set_trace()

        # create labels
        d_label = torch.tensor([self._data_label for i in range(batch_size)])
        g_label = torch.tensor([self._gen_label for i in range(batch_size)])


        d_data = self.discriminator(data)
        d_data_loss = self._loss(d_data, d_label)

        gen = self.generator()
        d_gen = self.discriminator(gen)
        d_gen_loss = self._loss(d_gen, g_label)

        d_loss = d_data_loss + d_gen_loss
        d_loss.backward(retain_graph=True)

        self._d_train.step()

        g_loss = self._loss(d_gen, d_label)
        g_loss.backward()
        self._g_train.step()

        return d_loss, g_loss


class Discriminiator(torch.nn.Module):
    HIDDEN_SIZE = 32

    def __init__(self, input_size):
        super(Discriminiator, self).__init__()

        self.lstm = torch.nn.LSTM(input_size, self.HIDDEN_SIZE, num_layers=1)
        self.final = torch.nn.Linear(self.HIDDEN_SIZE, 2)

    def forward(self, feed):
        # import ipdb; ipdb.set_trace()
        self._init_hidden_layer(feed.size(0))

        # first tranpose feed to the right dimension
        feed = feed.transpose(0, 1)
        # then apply lstm
        output, self.hidden = self.lstm(feed, self.hidden)
        # then apply final on the last output
        output = output.select(0, -1)
        output = self.final(output)

        return output

    def _init_hidden_layer(self, batch_size=1):
        def gen():
            return torch.randn(1, batch_size, self.HIDDEN_SIZE)
        self.hidden = (gen(), gen())


class Generator(torch.nn.Module):
    SEED_SIZE = 20
    OUTPUT_LENGTH = 160
    LSTM_OUTPUT_SIZE = 32

    def __init__(self, output_size, batch_size):
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size

        self.lstm = torch.nn.LSTM(self.SEED_SIZE, self.LSTM_OUTPUT_SIZE, num_layers=2)
        self.final = torch.nn.Linear(self.LSTM_OUTPUT_SIZE, output_size)

    def forward(self):
        self._init_hidden_layer()

        # make input to lstm
        feed = torch.randn(1, self.batch_size, Generator.SEED_SIZE)
        feed = feed.repeat(self.OUTPUT_LENGTH, 1, 1)

        output, self.hidden = self.lstm(feed, self.hidden)
        output = self.final(output)
        output = F.tanh(output)

        # format and return output
        output = output.transpose(0, 1)
        return output

    def _init_hidden_layer(self):
        def gen():
            return torch.zeros(2, self.batch_size, self.LSTM_OUTPUT_SIZE)
        self.hidden = (gen(), gen())


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # hyperparameters
    BATCH_SIZE = 10
    STEPS = 500

    # initialize data and loader
    data = data_loader.ChorData()
    data_loader = DataLoader(data,
                             batch_size=BATCH_SIZE,
                             sampler=data_loader.InfiniteSampler(data))

    # initialize models
    d = Discriminiator(data.input_size)
    g = Generator(data.input_size, BATCH_SIZE)
    gan = GAN(d, g)

    # train model
    dl = iter(data_loader)
    for i in range(STEPS):
        d_loss, g_loss = gan.train(next(dl), BATCH_SIZE)
        print(d_loss, g_loss)

    # print random picture from LSTM just for kicks
    import matplotlib.pyplot as plt
    output = g().select(0, 0)
    plt.matshow(output.detach().numpy())
    plt.show()
