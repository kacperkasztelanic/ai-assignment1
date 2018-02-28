import numpy as np
import random
import time

import data_loader as data_loader


class Simulation(object):
    POPULATIONS = 100
    GENERATIONS = 10
    PROBABILITY_BECOME_PARENT = 0.8
    PROBABILITY_DRAW = 1 - PROBABILITY_BECOME_PARENT
    PROBABILITY_MUTATE = 0.02

    def __init__(self, path):
        self.n, self.matrix_flow, self.matrix_distance = data_loader.load(path)
        self.assignments = []
        self.iter_fitness = []
        self.iter_costs = []
        self.dict_results = {}
        self.new_assignments = []
        self.prob_increasing = []
        self.no_populations = 0
        self.no_children = 0
        self.no_draw_parents = 0
        self.prob_mutate = 0
        self.fitness_fun_param = 0

    def predicate_fitness_fun_param(self):
        temp_asgn = []
        gen_costs = []
        for i in range(self.no_populations):
            temp_asgn.append(Assignment(size=self.n, m_flow=self.matrix_flow, m_dist=self.matrix_distance))
            temp_asgn[i].calc_fitness_and_cost_fun()
            gen_costs.append(temp_asgn[i].val_cost_fun)
        self.fitness_fun_param = int(max(gen_costs) * 1.1)
        print(self.fitness_fun_param)

    def get_data(self):
        return self.n, self.matrix_flow, self.matrix_distance

    def simulation(self, no_populations=POPULATIONS, no_generations=GENERATIONS, prob_parent=PROBABILITY_BECOME_PARENT,
                   prob_mutate=PROBABILITY_MUTATE):
        self.no_populations = no_populations
        self.no_children = int(prob_parent * no_populations)
        self.no_draw_parents = self.no_populations - self.no_children
        self.prob_mutate = prob_mutate
        self.predicate_fitness_fun_param()
        self.init_random_assignments()
        self.calc_fitness_and_cost_fun()
        self.update_dict(0)

        for g in range(1, no_generations, 1):
            self.reset_lists()
            self.calc_prob_increasing()
            self.selection_assignments_to_copy()
            self.crossover_children()
            self.assignments = self.new_assignments
            self.mutate_rand()
            self.calc_fitness_and_cost_fun()
            self.update_dict(g)
        return self.dict_results

    def init_random_assignments(self):
        for i in range(self.no_populations):
            self.assignments.append(Assignment(size=self.n, m_flow=self.matrix_flow, m_dist=self.matrix_distance))

    def calc_fitness_and_cost_fun(self):
        self.iter_fitness = []
        self.iter_costs = []
        for i in range(self.no_populations):
            self.assignments[i].calc_fitness_and_cost_fun(self.fitness_fun_param)
            self.iter_fitness.append(self.assignments[i].val_fitness_fun)
            self.iter_costs.append(self.assignments[i].val_cost_fun)

    def reset_lists(self):
        self.prob_increasing = []
        self.new_assignments = []

    def calc_prob_increasing(self):
        sum_val_fitness_fun = sum(self.iter_fitness)
        self.assignments[0].probability_reproduce = self.assignments[0].val_fitness_fun / sum_val_fitness_fun
        self.assignments[0].probability_increasing = self.assignments[0].probability_reproduce
        self.prob_increasing.append(self.assignments[0].probability_increasing)
        for i in range(1, self.no_populations, 1):
            self.assignments[i].probability_reproduce = self.assignments[i].val_fitness_fun / sum_val_fitness_fun
            self.assignments[i].probability_increasing = \
                self.assignments[i - 1].probability_increasing + self.assignments[i].probability_reproduce
            self.prob_increasing.append(self.assignments[i].probability_increasing)

    def crossover_children(self):
        for i in range(self.no_children):
            rand1 = random.random()
            rand2 = random.random()
            index1 = self.index_fst_elem_bigger(rand1, self.prob_increasing)
            index2 = self.index_fst_elem_bigger(rand2, self.prob_increasing)
            self.new_assignments.append(self.assignments[index1].crossover(self.assignments[index2]))

    def selection_assignments_to_copy(self):
        for i in range(self.no_draw_parents):
            rand_index = random.randint(0, self.no_populations - 1)
            self.new_assignments.append(self.assignments[rand_index])

    def mutate_rand(self):
        for i in range(self.no_draw_parents):
            if random.random() < self.prob_mutate:
                self.assignments[i].mutate()

    def update_dict(self, no_iter):
        self.dict_results.update({no_iter: [min(self.iter_costs), int(sum(self.iter_costs) / len(self.iter_costs)),
                                            max(self.iter_costs),
                                            list(filter(lambda x: x.val_cost_fun == min(self.iter_costs),
                                                        self.assignments))[0].factories]})

    @staticmethod
    def index_fst_elem_bigger(elem_to_check, data_list):
        return list(map(lambda x: x > elem_to_check, data_list)).index(True)


class Assignment(object):
    NUMBER_OF_DIVISIONS = 1
    NUMBER_OF_MUTATIONS = 1

    def __init__(self, size, m_flow, m_dist, factories=None, no_divisions=NUMBER_OF_DIVISIONS,
                 no_mutations=NUMBER_OF_MUTATIONS):
        self.size = size
        self.factories = factories
        self.val_fitness_fun = -1
        self.val_cost_fun = -1
        self.m_flow = m_flow
        self.m_dist = m_dist
        self.probability_reproduce = 0
        self.probability_increasing = 0
        self.no_divisions = no_divisions
        self.no_mutations = no_mutations
        if factories is None:
            self.factories = np.arange(self.size)
            self.random_assign()

    def random_assign(self):
        random.shuffle(self.factories)

    def mutate(self):
        for i in range(self.no_mutations):
            j = random.randrange(self.size)
            k = random.randrange(self.size)
            self.factories[j], self.factories[k] = self.factories[k], self.factories[j]

    # 0. 2 3. 6 1. 4 5   +
    # 5. 4 1. 6 4. 3 2   ==
    # 0. 4 1. 6 1. 3 2   repair
    # 0. 4 5. 6 1. 3 2
    def crossover(self, parent):
        divide_points = random.sample(range(self.size), self.no_divisions)
        divide_points.extend([0, self.size])
        divide_points.sort()
        dp = divide_points
        switch = True
        temp = np.array([], dtype=int)
        for i in range(self.no_divisions + 1):
            if switch:
                temp = np.append(temp, self.factories[dp[i]:dp[i + 1]])
            else:
                temp = np.append(temp, parent.factories[dp[i]:dp[i + 1]])
            switch = not switch
        child = Assignment(size=self.size, factories=temp, m_flow=self.m_flow, m_dist=self.m_dist)
        child.repair_genome()
        return child

    def repair_genome(self):
        temp = np.copy(self.factories)
        temp.sort()
        list_not_contains = []
        list_twice_contains = []
        for i in range(self.size - 1):
            if i not in temp:
                list_not_contains.append(i)
            if temp[i] == temp[i + 1]:
                list_twice_contains.append(temp[i])
        if not self.size - 1 in temp:
            list_not_contains.append(self.size - 1)
        random.shuffle(list_not_contains)
        for i in range(self.size - 1):
            if self.factories[i] in list_twice_contains:
                list_twice_contains.remove(self.factories[i])
                self.factories[i] = list_not_contains.pop(0)

    def calc_fitness_and_cost_fun(self, param=0):
        # size = len(self.factories)
        # new_m_dist = np.zeros(shape=(size,size))
        # for i in range(size):
        #     for j in range(size):
        #         new_m_dist[i][j] = self.m_dist[self.factories[i]][self.factories[j]]
        new_m_dist = self.m_dist[:, self.factories][self.factories]
        self.val_cost_fun = np.sum(np.multiply(self.m_flow, new_m_dist))
        if param - self.val_cost_fun > 0:
            self.val_fitness_fun = (param - self.val_cost_fun) * (param - self.val_cost_fun)
        else:
            self.val_fitness_fun = 0
            # self.val_fitness_fun = 1 / (self.val_cost_fun) / (self.val_cost_fun)
            # self.val_fitness_fun = 1 / (self.val_cost_fun - 6000) / (self.val_cost_fun - 6000)
            # self.val_fitness_fun = 1 / (self.val_cost_fun - 3000) / (self.val_cost_fun - 3000)


def main():
    start = time.time()

    # sim = Simulation('chr25a.dat')
    # sim = Simulation('chr12a.dat')
    # sim = Simulation('had12.dat')
    # sim = Simulation('had14.dat')
    # sim = Simulation('had16.dat')
    # sim = Simulation('had18.dat')
    sim = Simulation('had20.dat')
    test = sim.simulation(no_populations=100, no_generations=100)
    for d in test.items():
        print(d)

    # n, m_flow, m_dist = sim.get_data()
    # # print(sim.get_data())
    # best = Assignment(size=n, m_flow=m_flow, m_dist=m_dist)
    # # 'chr12a.dat'
    # # temp = np.array([7, 5, 12, 2, 1, 3, 9, 11, 10, 6, 8, 4])
    # # 'chr25a.dat'
    # # temp = np.array([25, 12, 5, 3, 18, 4, 16, 8, 20, 10, 14, 6, 15, 23, 24, 19, 13, 1, 21, 11, 17, 2, 22, 7, 9])
    # # 'chr22b.dat'
    # # temp = np.array([10, 19, 3, 1, 20, 2, 6, 4, 7, 8, 17, 12, 11, 15, 21, 13, 9, 5, 22, 14, 18, 16])
    # # best.factories = list(map(lambda x: x-1, temp))
    #
    # temp = np.array([8, 15, 16, 14, 19,  6,  7, 17,  1, 12, 10, 11,  5, 20,  2,  3,  4,  9, 18, 13])
    # best.factories = list(map(lambda x: x - 1, temp))
    # print(best.factories)
    # best.calc_fitness_and_val_cost_fun()
    # print(best.val_cost_fun)

    print("simulation time:")
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
