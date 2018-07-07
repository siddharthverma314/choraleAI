from torch.utils.data import DataLoader
import torch
import data_loader
import gan
import matplotlib.pyplot as plt


class Summary:
    def __init__(self, filepath):
        self.scalars = {}
        self.filepath = filepath

    def add_scalar(self, name, val):
        if name not in self.scalars:
            self.scalars[name] = [val.item()]
        else:
            self.scalars[name].append(val.item())

    def print_scalars(self, i):
        out = f"step: {i}, "
        for name in self.scalars:
            out += name + ": " + str(self.scalars[name][-1])
            out += ", "
        print(out[:-2])

    def save_scalars(self):
        for name in self.scalars:
            val = self.scalars[name]
            plt.plot(range(len(val)), val)
        plt.savefig(self.filepath + 'scalars.png')
        plt.close()

    def save_model(self, model, name, i):
        torch.save(model.state_dict(), f"{self.filepath}{name}_{i}.model")

    def save_img(self, tensor, name, i):
        plt.matshow(tensor.numpy())
        plt.savefig(f"{self.filepath}{name}_{i}.png")
        plt.close()


class Train:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._init_data()
        self._init_models()
        self.summ = Summary('./test/')

    def _init_data(self):
        self.data = data_loader.InterlaceChorData()
        self.dl = DataLoader(self.data,
                             batch_size=self.batch_size,
                             sampler=data_loader.InfiniteSampler(self.data))

    def _init_models(self):
        self.d = gan.Discriminiator(self.data.input_size)
        self.g = gan.Generator(self.data.input_size, self.batch_size)
        self.gan = gan.GAN(self.d, self.g)

    def train(self, train_steps=100000, save_step=100):
        di = iter(self.dl)
        for i in range(train_steps):
            d_loss, g_loss = self.gan.train(next(di), self.batch_size)

            self.summ.add_scalar('d_loss', d_loss)
            self.summ.add_scalar('g_loss', g_loss)

            self.summ.print_scalars(i)

            if i % save_step == 0:
                self.summ.save_model(self.d, "discriminator", i)
                self.summ.save_model(self.g, "generator", i)

                # save one image
                with torch.no_grad():
                    self.g.batch_size = 1
                    self.summ.save_img(self.g.forward().squeeze(), "gen_notes", i)
                    self.g.batch_size = self.batch_size

        self.summ.save_scalars()


if __name__ == "__main__":
    t = Train(5)
    t.train()
