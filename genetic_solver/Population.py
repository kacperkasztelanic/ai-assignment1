import numpy as np

from genetic_solver.Phenotype import Phenotype
from genetic_solver.SelectionType import SelectionType


class Population:
    def __init__(self, phenotype_size, flow_matrix, distance_matrix, population_size, crossover_prob, mutation_prob,
                 division_point_ratio, selection_type, tournament_size):
        self.phenotype_size = phenotype_size
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.division_point_ratio = division_point_ratio
        self.selection_type = selection_type
        self.tournament_size = tournament_size
        self.phenotypes = None
        self.new_phenotypes = None
        self.num_of_children = int(self.crossover_prob * self.population_size)
        self.num_of_parents_to_clone = self.population_size - self.num_of_children
        self.min_cost = None
        self.avg_cost = None
        self.max_cost = None
        self.prob_list = None
        self.fitness_values = None

    def generate_random_phenotypes(self):
        temp = [Phenotype(size=self.phenotype_size) for _ in range(self.population_size)]
        self.phenotypes = np.asarray(temp)
        self.calc_cost_and_fitness_functions()

    def get_next_population(self):
        self.new_phenotypes = []
        self.calc_prob_list()
        self.crossover()
        self.clone_parents_into_children()
        self.invoke_mutations()
        result = Population(phenotype_size=self.phenotype_size, flow_matrix=self.flow_matrix,
                            distance_matrix=self.distance_matrix, population_size=self.population_size,
                            crossover_prob=self.crossover_prob, mutation_prob=self.mutation_prob,
                            division_point_ratio=self.division_point_ratio, selection_type=self.selection_type,
                            tournament_size=self.tournament_size)
        result.phenotypes = self.new_phenotypes
        result.calc_cost_and_fitness_functions()
        return result

    def calc_cost_and_fitness_functions(self):
        cost_values = []
        self.fitness_values = []
        for p in self.phenotypes:
            p.calc_cost_function(flow_matrix=self.flow_matrix, distance_matrix=self.distance_matrix)
            cost_values.append(p.cost)
        self.fitness_values = np.asarray(cost_values)
        max_val = np.max(self.fitness_values) * 1.1
        temp = (max_val - self.fitness_values) / max_val
        self.fitness_values = np.multiply(max_val * temp, temp)

    def calc_prob_list(self):
        fitness_values_sum = np.sum(self.fitness_values)
        self.prob_list = self.fitness_values / fitness_values_sum

    def select_indices(self, n):
        if self.selection_type == SelectionType.ROULETTE:
            return self.select_roulette(n)
        elif self.selection_type == SelectionType.TOURNAMENT:
            return self.select_tournament(n)

    def select_roulette(self, n):
        return np.random.choice(self.population_size, size=n, replace=False, p=self.prob_list)

    def select_tournament(self, n):
        indices = np.random.randint(0, self.population_size, size=self.tournament_size * n)
        costs = [(self.phenotypes[indices[i]], indices[i]) for i in range(indices.shape[0])]
        costs.sort(key=lambda x: x[0].cost)
        return [costs[i][1] for i in range(n)]

    def crossover(self):
        for i in range(self.num_of_children):
            pair = self.select_indices(2)
            self.new_phenotypes.append(self.phenotypes[pair[0]].crossover(self.phenotypes[pair[1]], self.division_point_ratio))

    def clone_parents_into_children(self):
        for i in range(self.num_of_parents_to_clone):
            index = np.random.choice(self.population_size, size=1, p=self.prob_list)[0]
            self.new_phenotypes.append(self.phenotypes[index])

    def invoke_mutations(self):
        for p in self.new_phenotypes:
            p.mutate(self.mutation_prob)

    def get_results(self):
        temp = np.vectorize(lambda x: x.cost)(self.phenotypes)
        min_cost = np.min(temp)
        avg_cost = np.mean(temp)
        max_cost = np.max(temp)
        return np.array([min_cost, avg_cost, max_cost])
