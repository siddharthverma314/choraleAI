import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_FILE = './data/gan_data'


class Discriminiator(nn.Module):
    LSTM_INPUT_SIZE = 40
    HIDDEN_SIZE = 20
    LSTM_LAYERS = 1

    def __init__(self, input_size):
        super(Discriminiator, self).__init__()

        self.first = nn.Linear(input_size, self.LSTM_INPUT_SIZE)
        self.lstm = nn.LSTM(self.LSTM_INPUT_SIZE, self.HIDDEN_SIZE,
                            num_layers=self.LSTM_LAYERS)
        self.final = nn.Linear(self.HIDDEN_SIZE, 2)

    def forward(self, feed):
        self._init_hidden_layer(feed.size(0))

        # first tranpose feed to the right dimension
        feed = feed.transpose(0, 1)
        output = self.first(feed)
        output = F.relu(output)
        output, self.hidden = self.lstm(output, self.hidden)
        output = self.final(output)
        # output = output.select(0, -1)
        output = output.mean(0)

        return output

    def _init_hidden_layer(self, batch_size=1):
        def gen():
            return torch.randn(self.LSTM_LAYERS, batch_size, self.HIDDEN_SIZE)
        self.hidden = (gen(), gen())


class Generator(nn.Module):
    SEED_SIZE = 20
    OUTPUT_LENGTH = 50
    LSTM_OUTPUT_SIZE = 40
    LSTM_LAYERS = 2

    def __init__(self, output_size, batch_size):
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.SEED_SIZE, self.LSTM_OUTPUT_SIZE,
                            num_layers=self.LSTM_LAYERS)
        self.final = nn.Linear(self.LSTM_OUTPUT_SIZE, output_size)

    def forward(self):
        self._init_hidden_layer()

        # make input to lstm
        feed = torch.randn(1, self.batch_size, Generator.SEED_SIZE)
        feed = F.pad(feed, (0, 0, 0, 0, 0, self.OUTPUT_LENGTH - 1))
        # feed = feed.repeat(self.OUTPUT_LENGTH, 1, 1)

        output, self.hidden = self.lstm(feed, self.hidden)
        output = self.final(output)
        output = F.sigmoid(output)

        # format and return output
        output = output.transpose(0, 1)
        return output

    def _init_hidden_layer(self):
        def gen():
            return torch.rand(self.LSTM_LAYERS, self.batch_size, self.LSTM_OUTPUT_SIZE)
        self.hidden = (gen(), gen())


class GAN():

    def __init__(self, discriminator, generator):

        self.discriminator = discriminator
        self.generator = generator

        self._loss = F.cross_entropy

        self._data_label = 1
        self._gen_label = 0

        self._d_train = torch.optim.Adam(self.discriminator.parameters(), 0.01, weight_decay=0.1)
        self._g_train = torch.optim.Adam(self.generator.parameters(), 0.01, weight_decay=0.1)

    def train(self, data, batch_size):
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
