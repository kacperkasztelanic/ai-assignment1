import itertools
import math
import sys

import numpy as np


class BruteForceSolver:
    def __init__(self, n, flow_matrix, distance_matrix):
        self.n = n
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix

    def run(self):
        best_solution = None
        min_cost = None
        factorial = math.factorial(self.n)
        for i, permutation in zip(range(factorial), itertools.permutations(np.arange(self.n))):
            if i % (int(factorial / 10000000) + 1) == 0:
                sys.stdout.write("\r%4.5f%% %d / %d" % (i / factorial, i, factorial))
                sys.stdout.flush()
            permutation = np.array(permutation)
            cost = self.cost_function(permutation)
            if min_cost is None or cost < min_cost:
                best_solution = permutation
                min_cost = cost
        print()
        return min_cost, best_solution

    def cost_function(self, factories):
        new_distance_matrix = self.distance_matrix[:, factories][factories]
        return np.sum(np.multiply(self.flow_matrix, new_distance_matrix))
