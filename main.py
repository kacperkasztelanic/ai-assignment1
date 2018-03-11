import os

import matplotlib.pyplot as plt

from genetic_solver.SelectionType import SelectionType
from genetic_solver.Simulation import Simulation
from genetic_solver.SimulationRunner import SimulationRunner
from other_solvers import GreedySolver
from other_solvers import RandomSolver
from other_solvers.BruteForceSolver import BruteForceSolver
from utils.DataLoader import DataLoader
from utils.DataSaver import DataSaver
from utils.timing import timing

SOURCE_DIR = 'data'
SOURCE_EXT = 'dat'
SOLUTION_DIR = 'solutions'
SOLUTION_EXT = 'sln'
RESULTS_DIR = 'results'
RESULTS_EXT = 'csv'

RESULTS_FILENAME = 'results'

FILES = ['had12', 'had14', 'had16', 'had18', 'had20', 'had8']
FILE_INDEX = 4
ITERATIONS = 10
POPULATION_SIZE = 100
GENERATIONS = 200
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.04
SELECTION_TYPE = SelectionType.TOURNAMENT
TOURNAMENT_SIZE = 5
DIVISION_POINT_RATIO = 0.5

TIMES_RANDOM = 100000


def main():
    loader = DataLoader(source_dir=SOURCE_DIR, source_ext=SOURCE_EXT, solution_dir=SOLUTION_DIR, solution_ext=SOLUTION_EXT)
    n, flow_matrix, distance_matrix = loader.load_source(FILES[FILE_INDEX])
    optimal_sol = loader.load_results(FILES[FILE_INDEX])[1]

    results = genetic_solver(n, flow_matrix, distance_matrix)
    # optimal_sol = brute_force_solver(n, flow_matrix, distance_matrix)[0]
    random_res = random_solver(n, flow_matrix, distance_matrix, TIMES_RANDOM)[0]
    greedy_res = greedy_solver(n, flow_matrix, distance_matrix)

    save_results_to_csv(results)

    graph_filename = 'graph'
    graph_path = os.path.join(RESULTS_DIR, (graph_filename + '.png'))
    plot_graph(results, path=graph_path, random_res=random_res, greedy_res=greedy_res, optimal_sol=optimal_sol)


@timing
def genetic_solver(n, flow_matrix, distance_matrix):
    simulation = Simulation(n=n, flow_matrix=flow_matrix, distance_matrix=distance_matrix, population_size=POPULATION_SIZE,
                            generations=GENERATIONS, crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
                            division_point_ratio=DIVISION_POINT_RATIO, selection_type=SELECTION_TYPE,
                            tournament_size=TOURNAMENT_SIZE)
    multiple_sim = SimulationRunner(simulation=simulation, iterations=ITERATIONS)
    return multiple_sim.run()


@timing
def greedy_solver(n, flow_matrix, distance_matrix):
    greedy = GreedySolver(n=n, flow_matrix=flow_matrix, distance_matrix=distance_matrix)
    return greedy.run()


@timing
def brute_force_solver(n, flow_matrix, distance_matrix):
    brute_force = BruteForceSolver(n=n, flow_matrix=flow_matrix, distance_matrix=distance_matrix)
    return brute_force.run()


@timing
def random_solver(n, flow_matrix, distance_matrix, times):
    random = RandomSolver(n=n, flow_matrix=flow_matrix, distance_matrix=distance_matrix, times=times)
    return random.run()


def plot_graph(results, path=None, random_res=None, greedy_res=None, optimal_sol=None):
    x_max = results.shape[0] + 1
    x = range(1, x_max)
    marker = '.'
    e_line_width = 1
    capsize = 2
    axes = plt.gca()
    axes.set_xlim([1, x_max])
    plt.errorbar(x, list(results[:, 0]), list(results[:, 1]), marker=marker, elinewidth=e_line_width, capsize=capsize,
                 label='Min')
    plt.errorbar(x, list(results[:, 2]), list(results[:, 3]), marker=marker, elinewidth=e_line_width, capsize=capsize,
                 label='Avg')
    plt.errorbar(x, list(results[:, 4]), list(results[:, 5]), marker=marker, elinewidth=e_line_width, capsize=capsize,
                 label='Max')
    if random_res is not None:
        plt.axhline(y=random_res, color='m', linestyle='-', label=str('Random (cost: ' + str(random_res) + ')'))
    if greedy_res is not None:
        plt.axhline(y=greedy_res, color='r', linestyle='-', label=str('Greedy (cost: ' + str(greedy_res) + ')'))
    if optimal_sol is not None:
        plt.axhline(y=optimal_sol, color='c', linestyle='-', label=str('Optimal (cost: ' + str(optimal_sol) + ')'))
    plt.legend()
    plt.title('%s\nPop: %s | Px: %s | Pm: %s | %s | Tour: %s' %
              (FILES[FILE_INDEX].upper(), str(POPULATION_SIZE), str(CROSSOVER_PROB), str(MUTATION_PROB),
               SELECTION_TYPE.name.lower(), str(TOURNAMENT_SIZE)))
    plt.ylabel('Cost')
    plt.xlabel('Generation')
    if path is not None:
        plt.savefig(path)
    plt.show()


def save_results_to_csv(results):
    saver = DataSaver(results_dir=RESULTS_DIR, results_ext=RESULTS_EXT)
    saver.save(result=results, filename=RESULTS_FILENAME)


if __name__ == "__main__":
    main()
