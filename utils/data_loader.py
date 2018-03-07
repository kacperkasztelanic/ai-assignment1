import os

import numpy as np

SOURCE_DIR = 'data'
SOURCE_EXT = 'dat'
SOLUTION_DIR = 'solutions'
SOLUTION_EXT = 'sln'


def load_source(filename: str) -> (int, int, np.ndarray):
    path = os.path.join(SOURCE_DIR, (filename + '.' + SOURCE_EXT))
    with open(path, 'r') as data_file:
        n = int(data_file.readline())
        data_file.readline()
        matrix_flow = load_matrix(data_file, n)
        data_file.readline()
        matrix_distance = load_matrix(data_file, n)
    return n, matrix_flow, matrix_distance


def load_matrix(file, n: int) -> np.ndarray:
    matrix = ''
    for i in range(n):
        matrix += file.readline()
    return np.fromstring(matrix, dtype=int, sep=' ').reshape(n, n)


def load_results(filename: str) -> (int, int, np.ndarray):
    path = os.path.join(SOLUTION_DIR, (filename + '.' + SOLUTION_EXT))
    data_file = open(path, 'r')
    n, cost = data_file.readline().split()
    vector = np.fromstring(data_file.readline(), dtype=int, sep=' ')
    return int(n), int(cost), vector
