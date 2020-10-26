import argparse
import torch
import run
import encoding
from torch.utils.data import DataLoader
import torch.optim as optim
import data_loader as dl


class Train_VAE(run.Train):
    def __init__(self, batch_size, hidden_size=30):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        # model.to(device)

    def init_data(self):
        self.data = dl.InterlaceChorData(truncate=1)
        self.input_size = self.data.input_size
        self.dl = DataLoader(self.data,
                             batch_size=self.batch_size,
                             sampler=dl.InfiniteSampler(self.data))

    def init_models(self, alpha):
        self.model = encoding.VAE(self.input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def train(self, train_steps=10000, save_step=200):
        train_loss = torch.tensor(0)

        di = iter(self.dl)
        for i in range(train_steps):
            data = next(di)
            loss = self.model.train(data)

            self.train_loss += loss
            self.summ.add_scalar('loss', train_loss)

            if i % save_step == 0:
                self.summ.save_model(self.model, "VAE")

            self.summ.save_scalars()

            self.summ.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--save-step', default=1000, type=int)
    parser.add_argument('--save-filepath', type=str)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=0.001, type=float)
    parser.add_argument('--disable-cuda', type=bool)
    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    t = Train_VAE(args.batch_size)
    t.init_data()
    t.init_saving(args.save_filepath)
    t.init_models(args.alpha)
    t.train(args.steps + 1, args.save_step)
