import random
import torch
import subprocess as sp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import os


MIDI_LENGTH = 160
MIDI_FILTER = (30, 90)
DATA_FILE = 'gan_data.dat'
DATA_DIR = '../data/chorales/kern/'

######################
# KERN TO NOTE ARRAY #
######################


# main function to call from this module
def hum_encode_note_array(filepath, truncate=True):
    text = __get_file(filepath)

    if truncate:
        if __size(text) > MIDI_LENGTH:
            return

        array = np.zeros((MIDI_LENGTH, MIDI_FILTER[1] - MIDI_FILTER[0]))
    else:
        array = np.zeros((__size(text), MIDI_FILTER[1] - MIDI_FILTER[0]))

    for i, spine in enumerate(__get_spines(text)):
        __add_spine_to_array(spine, i, array)

    return array


def __size(text):
    count = 0
    for line in text.split('\n'):
        if line.strip():
            count += 1
    return count - 1


def __get_shell_output(cmd):
    output = sp.run(cmd, shell=True, stdout=sp.PIPE, universal_newlines=True)
    return output.stdout


def __get_file(filepath):
    output = __get_shell_output(f"cat {filepath} | notearray -t 8 -C --midi")
    return output


def __get_spines(text):
    text = text.split("\n")[1:-1]
    spines = np.zeros((len(text), 4), dtype=np.int32)

    for i, line in enumerate(text):
        line = line.split()
        spines[i, :] = list(map(int, line[4:]))

    return spines


def __add_spine_to_array(spine, pos, array):
    for num in spine:
        num = abs(num) - MIDI_FILTER[0]

        array[pos, num] = 1


def process_and_save_data(truncate=True):
    files = list(filter(lambda x: "krn" in x, os.listdir(DATA_DIR)))
    output = []

    for i, filepath in enumerate(files):
        na = hum_encode_note_array(DATA_DIR + filepath, truncate)
        if na is None:
            continue

        output.append(torch.FloatTensor(na))

    torch.save(output, DATA_FILE)


#################
# DATASET CLASS #
#################

class TruncatedChorData(Dataset):

    def __init__(self, truncate=160):
        self.truncate = truncate
        self.data = torch.load(DATA_FILE)

    def __getitem__(self, i):
        data = self.data[i]
        if len(data) > self.truncate:
            return data[:self.truncate, :]
        else:
            zeros = torch.zeros(self.truncate - data.size(0), data.size(1))
            return torch.cat((data, zeros))

    def __len__(self):
        return len(self.data)

    @property
    def input_size(self):
        return MIDI_FILTER[1] - MIDI_FILTER[0]


class InterlaceChorData(Dataset):

    def __init__(self, truncate=50):
        self.truncate = truncate
        self.data = torch.load(DATA_FILE)
        self._interlace()

    def _interlace(self):
        self.interlace = np.zeros(len(self.data), dtype=np.int32)
        for i, d in enumerate(self.data):
            self.interlace[i] = max((0, len(d) - self.truncate))
        for i in range(1, len(self.interlace)):
            self.interlace[i] += self.interlace[i-1]

    def __getitem__(self, i):
        sub = self.interlace - i
        for i, val in enumerate(sub):
            if val >= 0:
                break
        return self.data[i][val:val + self.truncate, :]

    def __len__(self):
        return self.interlace[-1]

    @property
    def input_size(self):
        return MIDI_FILTER[1] - MIDI_FILTER[0]


class InfiniteSampler(Sampler):

    def __init__(self, data_source):
        self.data = data_source

    def __iter__(self):
        return self

    def __next__(self):
        return random.randint(0, len(self.data) - 1)

    def __len__(self):
        return np.iinfo(np.int64).max


if __name__ == "__main__":
    import argparse

    # build command line parser
    parser = argparse.ArgumentParser(description="Create and test pytorch dataset.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', '-t', action='store_true', help="run tests")
    group.add_argument('--create', '-c', action='store_true', help="create dataset")
    parser.add_argument('--truncate', action='store_true', help="truncate dataset on creation")
    args = parser.parse_args()

    if args.create:
        process_and_save_data(args.truncate)

    elif args.test:
        import matplotlib.pyplot as plt

        d = InterlaceChorData()
        dl = DataLoader(d, batch_size=1, sampler=InfiniteSampler(d))
        dl = iter(dl)

        i = 0
        while True:
            p = next(dl)
            print(f"pos: {i}, size: {p.size()}")
            plt.matshow(p.squeeze())
            plt.show()
            i += 1
