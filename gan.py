import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DATA_FILE = './data/gan_data'


class GRU_Disc(nn.Module):
    LSTM_INPUT_SIZE = 20
    HIDDEN_SIZE = 10
    LSTM_LAYERS = 1

    def __init__(self, input_size):
        super(GRU_Disc, self).__init__()

        self.first = nn.Linear(input_size, self.LSTM_INPUT_SIZE)
        self.lstm = nn.GRU(input_size=self.LSTM_INPUT_SIZE,
                           hidden_size=self.HIDDEN_SIZE,
                           num_layers=self.LSTM_LAYERS,
                           batch_first=True)
        self.final = nn.Linear(self.HIDDEN_SIZE, 1)

    def forward(self, feed):
        self._init_hidden_layer(feed.size(0))

        output = Variable(feed)

        output = self.first(feed)
        output = F.elu(output)

        output, _ = self.lstm(output, self.hidden)

        output = output.select(1, -1)
        output = self.final(output)
        output = F.sigmoid(output)

        return output

    def _init_hidden_layer(self, batch_size=1):
        def gen():
            x = torch.rand(self.LSTM_LAYERS, batch_size, self.HIDDEN_SIZE)
            return Variable(x)
        self.hidden = gen()


class GRU_Gen(nn.Module):
    SEED_SIZE = 10
    LSTM_OUTPUT_SIZE = 15
    LSTM_LAYERS = 1

    def __init__(self, output_size, output_length, batch_size):
        super(GRU_Gen, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.output_length = output_length

        self.lstm = nn.GRU(input_size=self.SEED_SIZE,
                           hidden_size=self.LSTM_OUTPUT_SIZE,
                           num_layers=self.LSTM_LAYERS,
                           batch_first=True)
        self.final = nn.Linear(self.LSTM_OUTPUT_SIZE, output_size)

    def latent(self):
        return Variable(torch.randn(self.batch_size, 1, self.SEED_SIZE))

    def forward(self, feed):
        self._init_hidden_layer()

        # feed = F.pad(feed, (0, 0, 0, 0, 0, self.output_length - 1))
        feed = feed.repeat(1, self.output_length, 1)

        output, _ = self.lstm(feed, self.hidden)
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

    K = 0.01

    def __init__(self, discriminator, generator):

        self.disc = discriminator
        self.gen = generator

        self._data_label = 0
        self._gen_label = 1

        self._d_train = torch.optim.Adam(self.disc.parameters(), 0.01, weight_decay=0.2)
        self._g_train = torch.optim.Adam(self.gen.parameters(), 0.01, weight_decay=0.2)

    def train(self, data, batch_size, **kwargs):
        if "latent" not in kwargs:
            kwargs["latent"] = self.gen.latent()

        data = Variable(data)

        self.disc.zero_grad()
        self.gen.zero_grad()

        d_data = self.disc(data)
        d_gen = self.disc(self.gen(kwargs["latent"]))

        d_loss = torch.mean(torch.log(d_data + self.K) + torch.log(1 - d_gen + self.K))
        d_loss = -1 * d_loss  # change to gradient ascent
        d_loss.backward()
        self._d_train.step()

        self.disc.zero_grad()
        self.gen.zero_grad()

        d_gen = self.disc(self.gen(kwargs["latent"]))
        g_loss = torch.mean(torch.log(1 - d_gen + self.K))
        g_loss.backward()
        self._g_train.step()

        return d_loss, g_loss
