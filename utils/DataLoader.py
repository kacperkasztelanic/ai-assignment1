import os

import numpy as np


class DataLoader:
    def __init__(self, source_dir, source_ext, solution_dir, solution_ext):
        self.source_dir = source_dir
        self.source_ext = source_ext
        self.solution_dir = solution_dir
        self.solution_ext = solution_ext

    def load_source(self, filename: str) -> (int, int, np.ndarray):
        path = os.path.join(self.source_dir, (filename + '.' + self.source_ext))
        with open(path, 'r') as data_file:
            n = int(data_file.readline())
            data_file.readline()
            matrix_flow = self.load_matrix(data_file, n)
            data_file.readline()
            matrix_distance = self.load_matrix(data_file, n)
        return n, matrix_flow, matrix_distance

    def load_matrix(self, file, n: int) -> np.ndarray:
        matrix = ''
        for i in range(n):
            matrix += file.readline()
        return np.fromstring(matrix, dtype=int, sep=' ').reshape(n, n)

    def load_results(self, filename: str) -> (int, int, np.ndarray):
        path = os.path.join(self.solution_dir, (filename + '.' + self.solution_ext))
        data_file = open(path, 'r')
        n, cost = data_file.readline().split()
        vector = np.fromstring(data_file.readline(), dtype=int, sep=' ')
        return int(n), int(cost), vector
