import numpy as np


class RandomSolver(object):

    def __init__(self, n, flow_matrix, distance_matrix, times):
        self.n = n
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix
        self.times = times
        self.best_location = None
        self.min_cost = None

    def run(self):
        runs = [self.random_permutation_with_cost() for _ in range(self.times)]
        # cost_of_all = np.array([run[1] for run in runs])
        self.best_solution, self.cost = min(runs, key=lambda item: item[1])
        return self.cost

    def random_permutation_with_cost(self):
        locations = self.random_permutation(self.n)
        cost = self.cost_function(locations)
        return locations, cost

    def cost_function(self, factories):
        new_distance_matrix = self.distance_matrix[:, factories][factories]
        return np.sum(np.multiply(self.flow_matrix, new_distance_matrix))

    def random_permutation(self, n):
        result = np.arange(n)
        np.random.shuffle(result)
        return result
