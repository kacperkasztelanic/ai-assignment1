import numpy as np

import data_loader as loader
from Phenotype import Phenotype


class Simulation:
    POPULATION_SIZE = 200
    NUM_OF_GENERATIONS = 100
    PARENTING_RATIO = 0.8
    MUTATION_RATIO = 0.1

    def __init__(self, filename_root):
        self.n, self.flow_matrix, self.distance_matrix = loader.load_source(filename_root)
        self.optimal_cost = loader.load_results(filename_root)[1]
        self.phenotypes = []
        self.new_phenotypes = []
        self.num_of_generations = 0
        self.population_size = 0
        self.num_of_children = 0
        self.num_of_parents_to_clone = 0
        self.division_point_ratio = 0.5
        self.parenting_ratio = 0
        self.mutation_ratio = 0.02
        self.cost_values = []
        self.fitness_values = []
        self.prob_list = []
        self.fitness_function_param = 0
        self.results = []

    def run_simulation(self, population_size=POPULATION_SIZE, num_of_generations=NUM_OF_GENERATIONS,
                       parenting_ratio=PARENTING_RATIO, mutation_ratio=MUTATION_RATIO):
        self.population_size = population_size
        self.num_of_generations = num_of_generations
        self.parenting_ratio = parenting_ratio
        self.num_of_children = int(parenting_ratio * self.population_size)
        self.num_of_parents_to_clone = self.population_size - self.num_of_children
        self.mutation_ratio = mutation_ratio

        self.generate_random_phenotypes()
        self.calc_cost_and_fitness_functions()
        self.assumpt_fitness_function_param()
        self.calc_cost_and_fitness_functions()
        self.update_results()

        for i in range(self.num_of_generations):
            self.calc_prob_list()
            self.clone_parents_into_children()
            self.crossover()
            self.phenotypes = self.new_phenotypes
            self.new_phenotypes = []
            self.calc_cost_and_fitness_functions()
            self.update_results()
        self.reset()
        return self.results

    def generate_random_phenotypes(self):
        for i in range(self.population_size):
            self.phenotypes.append(Phenotype(self.n, self.flow_matrix, self.distance_matrix,
                                             division_point_ratio=self.division_point_ratio, mutation_prob=self.mutation_ratio))

    def calc_cost_and_fitness_functions(self):
        self.cost_values = []
        self.fitness_values = []
        for p in self.phenotypes:
            p.calc_cost_and_fitness_functions(self.fitness_function_param)
            self.cost_values.append(p.cost)
            self.fitness_values.append(p.fitness)

    def calc_prob_list(self):
        fitness_values_sum = sum(self.fitness_values)
        self.prob_list = [self.phenotypes[i].fitness / fitness_values_sum for i in range(self.population_size)]

    def crossover(self):
        for i in range(self.num_of_children):
            pair = np.random.choice(self.population_size, size=2, replace=False, p=self.prob_list)
            self.new_phenotypes.append(self.phenotypes[pair[0]].crossover(self.phenotypes[pair[1]]))

    def clone_parents_into_children(self):
        for i in range(self.num_of_parents_to_clone):
            index = np.random.choice(self.population_size, size=1, p=self.prob_list)[0]
            self.new_phenotypes.append(self.phenotypes[index])

    def invoke_mutations(self):
        for p in self.phenotypes:
            p.mutate()

    def update_results(self):
        length = len(self.cost_values)
        min_cost = min(self.cost_values)
        avg_cost = int(sum(list(map(lambda x: x / length, self.cost_values))))
        max_cost = max(self.cost_values)
        phenotype = np.ndarray.tolist(list(filter(lambda x: x.cost == min(self.cost_values),
                                                  self.phenotypes))[0].factories)
        self.results.append([min_cost, avg_cost, max_cost, phenotype])

    def assumpt_fitness_function_param(self):
        gen_costs = [self.phenotypes[i].cost for i in range(self.population_size)]
        self.fitness_function_param = int(max(gen_costs) * 1.1)

    def reset(self):
        self.phenotypes = []
        self.new_phenotypes = []
        self.num_of_generations = 0
        self.population_size = 0
        self.num_of_children = 0
        self.num_of_parents_to_clone = 0
        self.division_point_ratio = 0.5
        self.parenting_ratio = 0
        self.mutation_ratio = 0.02
        self.cost_values = []
        self.fitness_values = []
        self.prob_list = []
        self.fitness_function_param = 0
