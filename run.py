from torch.utils.data import DataLoader
import torch
import data_loader
import gan
import matplotlib.pyplot as plt
import argparse

class Summary:
    def __init__(self, filepath=None):
        self.scalars = {}
        self.filepath = filepath
        self.count = 0

    def add_scalar(self, name, val):
        if name not in self.scalars:
            self.scalars[name] = [val.item()]
        else:
            self.scalars[name].append(val.item())

    def print_scalars(self):
        out = f"step: {self.count}, "
        for name in self.scalars:
            out += name + ": " + str(self.scalars[name][-1])
            out += ", "
        print(out[:-2])

    def save_scalars(self, ylim=(-2, 2)):
        if self.filepath is None:
            return

        for name in self.scalars:
            val = self.scalars[name]
            plt.plot(range(len(val)), val)
            plt.ylim(*ylim)
        plt.savefig(self.filepath + 'scalars.png')
        plt.close()

    def save_model(self, model, name):
        if self.filepath is None:
            return

        filename = f"{self.filepath}{name}_{self.count}.model"
        torch.save(model.state_dict(), filename)

    def save_img(self, tensor, name):
        if self.filepath is None:
            return

        print(tensor.numpy())
        plt.matshow(tensor.numpy())
        filename = f"{self.filepath}{name}_{self.count}.png"
        plt.savefig(filename)
        plt.close()

    def step(self):
        self.count += 1


class Train:
    def __init__(self, batch_size, hidden_size=30):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.latent = None

    def init_data(self, length=10):
        self.output_length = length
        self.data = data_loader.InterlaceChorData(length)
        self.dl = DataLoader(self.data,
                             batch_size=self.batch_size,
                             sampler=data_loader.InfiniteSampler(self.data))

    def init_saving(self, directory):
        self.summ = Summary(directory)

    def init_models(self, prev_disc=None, prev_gen=None,
                    alpha=0.1, weight_decay=0.001):
        self.d = gan.GRU_Disc(self.data.input_size)
        self.g = gan.GRU_Gen(self.hidden_size, self.data.input_size)

        if prev_disc is not None:
            self.d.load_state_dict(torch.load(prev_disc))
        if prev_gen is not None:
            self.g.load_state_dict(torch.load(prev_gen))

        d_train = torch.optim.SGD(self.d.parameters(), alpha,
                                  weight_decay=weight_decay)
        g_train = torch.optim.SGD(self.g.parameters(), alpha,
                                  weight_decay=weight_decay)
        self.gan = gan.GAN(self.d, self.g, d_train, g_train)

    def train(self, train_steps=10000, save_step=200,
              d_train_steps=5, g_train_steps=5):
        const_latent = self.g.latent(1, self.output_length)
        di = iter(self.dl)
        d_loss, g_loss = torch.tensor(0), torch.tensor(0)
        for i in range(train_steps):
            data = next(di)
            latent = self.g.latent(self.batch_size, self.output_length)
            d_loss, g_loss = self.gan.train(data, latent, d_train_steps, g_train_steps)

            self.summ.add_scalar('d_loss', d_loss)
            self.summ.add_scalar('g_loss', g_loss)

            self.summ.print_scalars()

            if i % save_step == 0:
                self.summ.save_model(self.d, "discriminator")
                self.summ.save_model(self.g, "generator")

                # save one image
                with torch.no_grad():
                    out = self.g.forward(const_latent).squeeze(0)
                    self.summ.save_img(out, "gen_notes")

                self.summ.save_scalars()

            self.summ.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--save-step', default=1000, type=int)
    parser.add_argument('--prev-disc', type=str)
    parser.add_argument('--prev-gen', type=str)
    parser.add_argument('--save-filepath', type=str)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=0.001, type=float)
    parser.add_argument('--disable-cuda', type=bool)
    args = parser.parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    t = Train(args.batch_size)
    t.init_data()
    t.init_saving(args.save_filepath)
    t.init_models(args.prev_disc, args.prev_gen, args.alpha, args.weight_decay)
    t.train(args.steps + 1, args.save_step)
