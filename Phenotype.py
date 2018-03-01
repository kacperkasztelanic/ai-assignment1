import numpy as np

class Phenotype:
    def __init__(self, size, flow_matrix, distance_matrix, factories = None, division_point_ratio=0.5, mutation_prob = 0.2):
        self.size = size
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix
        self.division_point_ratio = division_point_ratio
        self.division_point = int(self.size * division_point_ratio)
        self.mutation_prob = mutation_prob
        self.factories = factories
        if self.factories is None:
            self.factories = np.arange(self.size)
            np.random.shuffle(self.factories)

    def crossover(self, partner):
        temp = self.factories[0:self.division_point]
        temp = np.append(temp, partner.factories[partner.division_point:])
        child = Phenotype(self.size, self.flow_matrix, self.distance_matrix, temp, self.division_point_ratio)
        child.genome_repair()
        return child

    def mutate(self):
        for i in range(self.size):
            if np.random.uniform(0, 1) <= self.mutation_prob:
                j = np.random.randint(0, self.size)
                self.factories[i], self.factories[j] = self.factories[j], self.factories[i]

    def genome_repair(self):
        temp = np.copy(self.factories)
        temp.sort()
        absent_genes = []
        duplicated_genes = []
        for i in range(self.size - 1):
            if i not in temp:
                absent_genes.append(i)
            if temp[i] == temp[i + 1]:
                duplicated_genes.append(temp[i])
        if self.size - 1 not in temp:
            absent_genes.append(self.size - 1)
        np.random.shuffle(absent_genes)
        for i in range(self.size - 1):
            if self.factories[i] in duplicated_genes:
                duplicated_genes.remove(self.factories[i])
                self.factories[i] = absent_genes.pop(0)
