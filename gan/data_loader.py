import subprocess as sp
import torch
from torch.utils.data import Dataset
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
def hum_encode_note_array(filepath):
    text = __get_file(filepath)

    if __size(text) > MIDI_LENGTH:
        return

    array = np.zeros((MIDI_LENGTH, MIDI_FILTER[1] - MIDI_FILTER[0]))

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


def process_and_save_data():
    files = list(filter(lambda x: "krn" in x, os.listdir(DATA_DIR)))
    output = []

    for i, filepath in enumerate(files):
        na = hum_encode_note_array(DATA_DIR + filepath)
        if na is None:
            continue

        output.append(torch.FloatTensor(na))

    torch.save(output, DATA_FILE)


#################
# DATASET CLASS #
#################

class ChorData(Dataset):

    def __init__(self):
        self.data = torch.load(DATA_FILE)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    @property
    def input_size(self):
        return MIDI_FILTER[1] - MIDI_FILTER[0]


if __name__ == "__main__":
    process_and_save_data()
