import numpy as np

import data_loader as loader
from Population import Population
from Population import SelectionType


class Simulation:
    POPULATION_SIZE = 100
    GENERATIONS = 100
    CROSSOVER_PROB = 0.8
    MUTATION_PROB = 0.4
    SELECTION_TYPE = SelectionType.TOURNAMENT
    TOURNAMENT_SIZE = 10
    DIVISION_POINT_RATIO = 0.5

    def __init__(self, filename_root, population_size=POPULATION_SIZE, generations=GENERATIONS, crossover_prob=CROSSOVER_PROB,
                 mutation_prob=MUTATION_PROB, division_point_ratio=DIVISION_POINT_RATIO, selection_type=SELECTION_TYPE,
                 tournament_size=TOURNAMENT_SIZE):
        self.filename_root = filename_root
        self.n, self.flow_matrix, self.distance_matrix = loader.load_source(self.filename_root)
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
        self.populations = [Population(phenotype_size=self.n, flow_matrix=self.flow_matrix, distance_matrix=self.distance_matrix,
                                       population_size=self.population_size, crossover_prob=self.crossover_prob,
                                       mutation_prob=self.mutation_prob, division_point_ratio=self.division_point_ratio,
                                       selection_type=self.selection_type, tournament_size=self.tournament_size)]
        self.populations[0].generate_random_phenotypes()
        self.update_results(0)

        for i in range(1, self.generations):
            self.populations.append(self.populations[i - 1].get_next_population())
            self.update_results(i)
        return self.results

    def update_results(self, i):
        self.results[i] = self.populations[i].get_results()
