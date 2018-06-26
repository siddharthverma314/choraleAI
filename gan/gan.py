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

        self._loss = lambda x, y: torch.dot(y, torch.log(x))

        self._data_label = torch.tensor([0, 1])
        self._gen_label = torch.tensor([0, 1])

        self._d_train = torch.optim.Adam(self.discriminator.parameters())
        self._g_train = torch.optim.Adam(self.generator.parameters())

    def train(self, data):
        # create generated image
        self.discriminator.zero_grad()
        self.generator.zero_grad()

        gen = self.generator()

        # import ipdb; ipdb.set_trace()
        d_data_loss = self._loss(self.discriminator(data), self._data_label)
        d_gen_loss = self._loss(self.discriminator(gen), self._train_label)
        d_loss = d_data_loss + d_gen_loss
        d_loss.backward()
        self._d_train.step()

        g_loss = self._loss(self.discriminator(gen), self._data_label)
        g_loss.backward()
        self._g_train.step()


class Discriminiator(torch.nn.Module):
    HIDDEN_SIZE = 128

    def __init__(self, size):
        super(Discriminiator, self).__init__()

        self.lstm = torch.nn.LSTM(size, self.HIDDEN_SIZE)
        self.final = torch.nn.Linear(self.HIDDEN_SIZE, 2)

    def forward(self, input):
        self._init_hidden_layer()

        for i in range(input.size(1)):
            feed = input[:, :, i]
            output, self.hidden = self.lstm(feed, self.hidden)

        return self.final(output)

    def _init_hidden_layer(self):
        def gen():
            return torch.randn(1, 1, self.HIDDEN_SIZE)
        self.hidden = (gen(), gen())


class Generator(torch.nn.Module):
    SEED_SIZE = 20
    OUTPUT_LENGTH = 160

    def __init__(self, size, batch_size):
        super(Generator, self).__init__()

        self.lstm = torch.nn.LSTM(self.SEED_SIZE, self.size)
        self.size = size

    def forward(self, batch):
        # multiple generate
        pass

    def _generate(self):
        input = torch.randn(1, 1, Generator.SEED_SIZE)
        self._init_hidden_layer()

        output = torch.zeros(self.size, self.OUTPUT_LENGTH)

        for i in range(self.OUTPUT_LENGTH):
            out, self.hidden = self.lstm(input, self.hidden)
            output[:, i] = out.view(self.size)

        return output

    def _init_hidden_layer(self):
        def gen():
            return torch.randn(1, 1, self.size)
        self.hidden = (gen(), gen())


if __name__ == "__main__":
    data = data_loader.ChorData()
    import ipdb; ipdb.set_trace()
    d = Discriminiator()
    print(d.forward(data[0]))
    # g = Generator()
    # g = GAN(d, g)
    # g.train(data[0])
