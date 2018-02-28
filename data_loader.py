import numpy as np


def load(filename: str):

    path = 'data/' + filename
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

