import numpy as np

from utils import data_saver as saver


class SimulationRunner:
    def __init__(self, simulation, results_filename, iterations):
        self.simulation = simulation
        self.iterations = iterations
        self.arr_results = np.empty(shape=(self.simulation.generations, 3, self.iterations))
        self.results_min_avg_max_std = np.empty(shape=(self.simulation.generations, 6))
        self.results_filename = results_filename

    def run_simulation(self):
        for i in range(0, self.iterations):
            self.simulation.run()
            self.arr_results[:, :, i] = self.simulation.results
        min_avg_max = np.sum(self.arr_results, axis=2) / self.iterations
        std = np.std(self.arr_results, axis=2)
        self.results_min_avg_max_std[:, :3] = min_avg_max
        self.results_min_avg_max_std[:, 3:] = std
        self.results_min_avg_max_std = self.results_min_avg_max_std[:, np.array([0, 3, 1, 4, 2, 5])]
        # print(self.results_min_avg_max_std)
        saver.save(result=self.results_min_avg_max_std, filename=self.results_filename)
