import torch
import matplotlib.pyplot as plt


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
    def init_data(self):
        raise NotImplementedError

    def init_saving(self, directory):
        self.summ = Summary(directory)

    def init_models(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
