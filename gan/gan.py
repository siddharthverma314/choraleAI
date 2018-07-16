import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DATA_FILE = './data/gan_data'


class Discriminiator(nn.Module):
    LSTM_INPUT_SIZE = 60
    HIDDEN_SIZE = 20
    LSTM_LAYERS = 1

    def __init__(self, input_size):
        super(Discriminiator, self).__init__()

        # self.first = nn.Linear(input_size, self.LSTM_INPUT_SIZE)
        self.lstm = nn.GRU(input_size=self.LSTM_INPUT_SIZE,
                           hidden_size=self.HIDDEN_SIZE,
                           num_layers=self.LSTM_LAYERS)
        self.final = nn.Linear(self.HIDDEN_SIZE, 2)

    def forward(self, feed):
        self._init_hidden_layer(feed.size(0))

        # first tranpose feed to the right dimension
        output = Variable(feed.transpose(0, 1))
        # output = self.first(feed)
        # output = F.relu(output)
        output, _ = self.lstm(output, self.hidden)
        output = output.select(0, -1)
        output = self.final(output)

        return output

    def _init_hidden_layer(self, batch_size=1):
        def gen():
            x = torch.rand(self.LSTM_LAYERS, batch_size, self.HIDDEN_SIZE)
            return Variable(x)
        self.hidden = gen()


class Generator(nn.Module):
    SEED_SIZE = 20
    LSTM_OUTPUT_SIZE = 40
    LSTM_LAYERS = 1

    def __init__(self, output_size, output_length, batch_size):
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.output_length = output_length

        self.lstm = nn.GRU(input_size=self.SEED_SIZE,
                           hidden_size=self.LSTM_OUTPUT_SIZE,
                           num_layers=self.LSTM_LAYERS)
        self.final = nn.Linear(self.LSTM_OUTPUT_SIZE, output_size)

    def forward(self):
        self._init_hidden_layer()

        # make input to lstm
        feed = torch.randn(1, self.batch_size, Generator.SEED_SIZE)
        # feed = F.pad(feed, (0, 0, 0, 0, 0, self.output_length - 1))
        feed = feed.repeat(self.output_length, 1, 1)
        feed = Variable(feed)

        output, self.hidden = self.lstm(feed, self.hidden)
        output = self.final(output)
        output = F.sigmoid(output)

        # format and return output
        output = output.transpose(0, 1)
        return output

    def _init_hidden_layer(self):
        def gen():
            x = torch.rand(self.LSTM_LAYERS, self.batch_size, self.LSTM_OUTPUT_SIZE)
            return Variable(x)
        self.hidden = gen()


class GAN():

    def __init__(self, discriminator, generator):

        self.discriminator = discriminator
        self.generator = generator

        self._loss = F.cross_entropy

        self._data_label = 0
        self._gen_label = 1

        self._d_train = torch.optim.SGD(self.discriminator.parameters(), 0.008, weight_decay=0.2)
        self._g_train = torch.optim.SGD(self.generator.parameters(), 0.008, weight_decay=0.2)

    def train(self, data, batch_size):
        data = Variable(data)

        self.discriminator.zero_grad()
        self.generator.zero_grad()

        d_label = Variable(torch.tensor(self._data_label).repeat(batch_size))
        g_label = Variable(torch.tensor(self._gen_label).repeat(batch_size))

        d_data = self.discriminator(data)
        d_data_loss = self._loss(d_data, d_label)
        d_data_loss.backward()

        gen = self.generator()
        d_gen = self.discriminator(gen)
        d_gen_loss = self._loss(d_gen, g_label)
        d_gen_loss.backward()

        d_loss = d_data_loss + d_gen_loss

        self._d_train.step()

        self.discriminator.zero_grad()
        self.generator.zero_grad()

        d_gen = self.discriminator(self.generator())
        g_loss = self._loss(d_gen, d_label)
        g_loss.backward()
        self._g_train.step()

        return d_loss, g_loss
