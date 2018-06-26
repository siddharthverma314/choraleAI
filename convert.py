import numpy as np
import os

# globals
PARENT_DIR = './data/chorales/kern/'


######################
# KERN TO NOTE ARRAY #
######################

# main function to call from this module
def hum_encode_note_array(filepath):
    text = __get_file(filepath)

    array = __create_blank_array(__size(text))

    for spine in __get_spines(text):
        __add_spine_to_array(spine, array)

    return array


def __size(text):
    count = 0
    for line in text.split('\n'):
        if line.strip():
            count += 1
    return count - 1


def __create_blank_array(size):
    return np.zeros((size, 128))


def __get_file(filepath):
    os.system(f"cat {filepath} | notearray -t 8 -C --midi > temp")
    output = open('temp').read()
    os.system("rm temp")
    return output


def __get_spines(text):
    spines = []
    for i in [5, 6, 7, 8]:
        command = "echo '%s' | awk '{print $%d}' > temp" % (text, i)
        os.system(command)
        spine = open('temp').read()
        os.system("rm temp")

        spine = spine.split("\n", 1)[1]
        spines.append(spine)
    return spines


def __add_spine_to_array(spine, array):
    count = 0

    for num in spine.strip().split('\n'):
        if num[0] == '-':
            num = int(num[1:])
        else:
            num = int(num)

        array[count, num] = 1
        count += 1


#########################
# KERN CHARACTER PARSER #
#########################

# build set of all characters
def __build_chars():

    chars = set()

    files = os.listdir(PARENT_DIR)
    for file in files:
        with open(PARENT_DIR + file) as f:
            text = f.read()
            for char in text:
                chars.add(char)

    return ''.join(list(sorted(chars)))


CHARS = __build_chars()


def __create_one_shot(char=None):
    index = CHARS.index(char)
    return [0 if i != index else 1 for i in range(len(CHARS))]


def hum_encode_character(filepath):
    with open(filepath) as f:
        string = f.read()

    encoding = []

    for i in range(len(string)):
        one_shot = __create_one_shot(string[i])
        encoding.append(one_shot)

    return encoding


def hum_decode_character(encoding):
    encoding = encoding.copy()
    output = ""

    while len(encoding) > 0:
        one_shot = encoding.pop()
        char = CHARS[one_shot.index(1)]
        output = char + output
    return output


##################
# CHORALE PARSER #
##################

# needed for the parser since strings are immutable
class ListIterator:
    def __init__(self, string):
        self.string = string
        self.index = 0

    def empty(self):
        return self.index == len(self.string)

    def poll(self):
        item = self.string[self.index]
        self.index += 1
        return item

    def peek(self):
        return self.string[self.index]


def parse_chorale(string):
    return __parse_string(string)


def __parse_string(string):
    stack = [[]]

    lst = ListIterator(string)

    while not lst.empty():
        if lst.peek() == ' ':
            lst.poll()
        elif lst.peek() == "(":
            lst.poll()
            stack.append([])
        elif lst.peek() == ")":
            lst.poll()
            elem = stack.pop()
            stack[-1].append(elem)
        else:
            stack[-1].append(__parse_atom(lst))

    return stack[0][0]


def __parse_atom(lst):
    atom = ""
    while not lst.empty() and lst.peek() not in [')', ' ']:
        atom += lst.poll()

    if atom.isnumeric():
        return int(atom)
    return atom


class ChoraleReader:
    def __init__(self, fp):
        self.f = open(fp, 'r')
        self.__curline = ""

    def __iter__(self):

        def get_lines():
            while True:
                line = self.f.readline()
                if line.strip():
                    yield self.process_line(line)

        return get_lines()

    def process_line(self, line):
        return parse_chorale(line)
