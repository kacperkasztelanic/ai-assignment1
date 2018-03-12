import numpy as np


class SimulationRunner:
    def __init__(self, simulation, iterations):
        self.simulation = simulation
        self.iterations = iterations
        self.arr_results = np.empty(shape=(self.simulation.generations, 3, self.iterations))
        self.results_min_avg_max_std = np.empty(shape=(self.simulation.generations, 6))

    def run(self):
        for i in range(0, self.iterations):
            self.simulation.run()
            self.arr_results[:, :, i] = self.simulation.results
        min_avg_max = np.sum(self.arr_results, axis=2) / self.iterations
        std = np.std(self.arr_results, axis=2)
        self.results_min_avg_max_std[:, :3] = min_avg_max
        self.results_min_avg_max_std[:, 3:] = std
        self.results_min_avg_max_std = self.results_min_avg_max_std[:, np.array([0, 3, 1, 4, 2, 5])]
        index_of_iter_with_lowest_end_cost = np.argmin(self.arr_results[-1, 0, :])
        best_iter = self.arr_results[:, 0, index_of_iter_with_lowest_end_cost]
        return np.around(self.results_min_avg_max_std, decimals=2), best_iter
