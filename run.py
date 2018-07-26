from torch.utils.data import DataLoader
import torch
import data_loader
import gan
import matplotlib.pyplot as plt


class Summary:
    def __init__(self, filepath):
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
        for name in self.scalars:
            val = self.scalars[name]
            plt.plot(range(len(val)), val)
            plt.ylim(*ylim)
        plt.savefig(self.filepath + 'scalars.png')
        plt.close()

    def save_model(self, model, name):
        torch.save(model.state_dict(), f"{self.filepath}{name}_{self.count}.model")

    def save_img(self, tensor, name):
        print(tensor.numpy())
        plt.matshow(tensor.numpy())
        plt.savefig(f"{self.filepath}{name}_{self.count}.png")
        plt.close()

    def step(self):
        self.count += 1


class Train:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._init_data()
        self._init_models()
        self.summ = Summary('./test/')
        self.latent = None

    def _init_data(self):
        self.data = data_loader.InterlaceChorData(1)
        self.dl = DataLoader(self.data,
                             batch_size=self.batch_size,
                             sampler=data_loader.InfiniteSampler(self.data))

    def _init_models(self):
        self.d = gan.ANN_Disc(self.data.input_size)
        self.g = gan.ANN_Gen(5, self.data.input_size, 1)
        d_train = torch.optim.SGD(self.d.parameters(), 0.02, weight_decay=0.1)
        g_train = torch.optim.SGD(self.g.parameters(), 0.02, weight_decay=0.1)
        self.gan = gan.GAN(self.d, self.g, d_train, g_train)

    def train(self, train_steps=10000, save_step=200):
        const_latent = self.g.latent(1)
        di = iter(self.dl)
        d_loss, g_loss = torch.tensor(0), torch.tensor(0)
        for i in range(train_steps):
            data = next(di)
            latent = self.g.latent(self.batch_size)
            d_loss, g_loss = self.gan.train(data, latent)

            self.summ.add_scalar('d_loss', d_loss)
            self.summ.add_scalar('g_loss', g_loss)

            self.summ.print_scalars()

            if i % save_step == 0:
                self.summ.save_model(self.d, "discriminator")
                self.summ.save_model(self.g, "generator")

                # save one image
                with torch.no_grad():
                    self.summ.save_img(self.g.forward(const_latent).squeeze(0), "gen_notes")

                self.summ.save_scalars()

            self.summ.step()


if __name__ == "__main__":
    t = Train(10)
    t.train(100001, 1000)
