import numpy as np

SOURCE_DIR = 'data'
SOURCE_EXT = 'dat'
SOLUTION_DIR = 'solutions'
SOLUTION_EXT = 'sln'


def load_source(filename: str):
    path = SOURCE_DIR + '/' + filename + '.' + SOURCE_EXT
    data_file = open(path, 'r')
    n = int(data_file.readline())
    data_file.readline()
    matrix = ''
    for i in range(n):
        matrix += data_file.readline()
    matrix_flow = np.fromstring(matrix, dtype=int, sep=' ').reshape(n, n)
    data_file.readline()
    matrix = ''
    for i in range(n):
        matrix += data_file.readline()
    matrix_distance = np.fromstring(matrix, dtype=int, sep=' ').reshape(n, n)
    return n, matrix_flow, matrix_distance


def load_results(filename: str):
    path = SOLUTION_DIR + '/' + filename + '.' + SOLUTION_EXT
    data_file = open(path, 'r')
    n, cost = data_file.readline().split()
    vector = np.fromstring(data_file.readline(), dtype=int, sep=' ')
    return int(n), int(cost), vector
