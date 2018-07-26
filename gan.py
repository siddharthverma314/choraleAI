import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_FILE = './data/gan_data'


def make_gru_hidden(gru_layers, batch_size, hidden_size):
    hidden = torch.rand(gru_layers, batch_size, hidden_size)
    return hidden


class GRU_Disc(nn.Module):

    def __init__(self, input_size, hidden_size=10, gru_layers=1):
        super(GRU_Disc, self).__init__()

        self.gru_layers = gru_layers
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=gru_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, feed):
        h = make_gru_hidden(self.gru_layers, feed.size(0), self.hidden_size)
        out = feed
        out, _ = self.rnn(out, h)
        out = out.select(1, -1)
        out = self.linear(out)
        out = F.sigmoid(out)
        return out


class GRU_Gen(nn.Module):

    def __init__(self, input_size, output_size, gru_layers=1):
        super(GRU_Gen, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gru_layers = gru_layers

        self.rnn = nn.GRU(input_size=input_size, hidden_size=output_size,
                          num_layers=gru_layers, batch_first=True)

    def latent(self, batch_size, output_length, fill_zeros=False):
        feed = torch.randn(batch_size, 1, self.input_size)
        if fill_zeros:
            feed = F.pad(feed, (0, 0, 0, output_length - 1, 0, 0))
        else:
            feed = feed.repeat(1, output_length, 1)
        return feed

    def forward(self, feed):
        h = make_gru_hidden(self.gru_layers, feed.size(0), self.output_size)
        out, _ = self.rnn(feed, h)
        # scale output between 0 and 1
        out = (out + 1) / 2
        return out


class ANN_Disc(nn.Module):

    def __init__(self, input_size, hidden_size=30):
        super(ANN_Disc, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, feed):
        feed = feed.view(feed.size(0), feed.size(1)*feed.size(2))
        out = feed
        out = self.l1(out)
        out = F.sigmoid(out)
        out = self.l2(out)
        out = F.sigmoid(out)
        return out


class ANN_Gen(nn.Module):

    def __init__(self, input_size, output_size, output_length, hidden_size=30):
        super(ANN_Gen, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.output_length = output_length
        self.input_size = input_size

    def latent(self, batch_size):
        feed = torch.rand(batch_size, self.input_size)
        return feed

    def forward(self, feed):
        out = feed
        out = self.l1(out)
        out = F.sigmoid(out)
        out = self.l2(out)
        out = F.sigmoid(out)
        out = out.view(out.size(0), self.output_length,
                       out.size(1)//self.output_length)
        return out


class GAN():

    def __init__(self, disc, gen, d_train, g_train):

        self.disc = disc
        self.gen = gen

        self.d_train = d_train
        self.g_train = g_train

    def train(self, data, latent, train_disc=True, train_gen=True,
              d_train_steps=5, g_train_steps=5):
        def lg(x):
            return torch.log(x + 1e-300)

        d_loss, g_loss = None, None  # initialize so at least None is returned

        if train_disc:
            for _ in range(d_train_steps):
                self.disc.zero_grad()
                self.gen.zero_grad()

                d_data = self.disc(data)
                d_gen = self.disc(self.gen(latent))

                d_loss = torch.mean(lg(d_data) + lg(1 - d_gen))
                d_loss = -1 * d_loss  # change to gradient ascent
                d_loss.backward()
                self.d_train.step()

        if train_gen:
            for _ in range(g_train_steps):
                self.disc.zero_grad()
                self.gen.zero_grad()

                d_gen = self.disc(self.gen(latent))
                g_loss = torch.mean(lg(1 - d_gen))
                g_loss.backward()
                self.g_train.step()

        return d_loss, g_loss



