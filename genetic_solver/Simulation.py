import numpy as np

from genetic_solver.Population import Population


class Simulation:
    def __init__(self, n, flow_matrix, distance_matrix, population_size, generations, crossover_prob, mutation_prob,
                 division_point_ratio, selection_type, tournament_size):
        self.n = n
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_type = selection_type
        self.tournament_size = tournament_size
        self.division_point_ratio = division_point_ratio
        self.populations = None
        self.results = np.empty(shape=(self.generations, 3))

    def run(self):
        self.populations = [Population(genotype_size=self.n, flow_matrix=self.flow_matrix, distance_matrix=self.distance_matrix,
                                       population_size=self.population_size, crossover_prob=self.crossover_prob,
                                       mutation_prob=self.mutation_prob, division_point_ratio=self.division_point_ratio,
                                       selection_type=self.selection_type, tournament_size=self.tournament_size)]
        self.populations[0].generate_random_genotypes()
        self.update_results(0)

        for i in range(1, self.generations):
            self.populations.append(self.populations[i - 1].get_next_population())
            self.update_results(i)
        return self.results

    def update_results(self, i):
        self.results[i] = self.populations[i].get_results()
