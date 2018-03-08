import numpy as np


class Phenotype:
    def __init__(self, size, factories=None):
        self.size = size
        self.factories = factories
        self.cost = None
        if self.factories is None:
            self.factories = np.arange(self.size)
            np.random.shuffle(self.factories)

    def crossover(self, partner, division_point_ratio):
        division_point = int(self.size * division_point_ratio)
        temp = self.factories[0:division_point]
        temp = np.append(temp, partner.factories[division_point:])
        child = Phenotype(self.size, temp)
        child.genome_repair()
        return child

    def mutate(self, mutation_prob):
        for i in range(self.size):
            if np.random.uniform(0, 1) <= mutation_prob:
                j = np.random.randint(0, self.size)
                self.factories[i], self.factories[j] = self.factories[j], self.factories[i]
        return self

    def genome_repair(self):
        counts = np.bincount(self.factories, minlength=self.size)
        absent_genes = np.core.numeric.where(counts == 0)[0]
        duplicated_genes = np.where(counts == 2)[0]
        for replace_to, replace_from in zip(absent_genes, duplicated_genes):
            self.factories[np.where(self.factories == replace_from)[0][0]] = replace_to

    def genome_repair_slow(self):
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

    def calc_cost_function(self, flow_matrix, distance_matrix):
        new_distance_matrix = distance_matrix[:, self.factories][self.factories]
        self.cost = np.sum(np.multiply(flow_matrix, new_distance_matrix))
