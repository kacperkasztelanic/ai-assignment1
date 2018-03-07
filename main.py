from SelectionType import SelectionType
from Simulation import Simulation
from SimulationRunner import SimulationRunner
from timing import timing

FILES = ['had12', 'had14', 'had16', 'had18', 'had20']
FILE_INDEX = 4
RESULTS_FILENAME = 'results.csv'
ITERATIONS = 10
POPULATION_SIZE = 100
GENERATIONS = 200
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.02
SELECTION_TYPE = SelectionType.TOURNAMENT
TOURNAMENT_SIZE = 10
DIVISION_POINT_RATIO = 0.5


@timing
def main():
    simulation = Simulation(filename_root=FILES[FILE_INDEX], population_size=POPULATION_SIZE, generations=GENERATIONS,
                            crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB, division_point_ratio=DIVISION_POINT_RATIO,
                            selection_type=SELECTION_TYPE, tournament_size=TOURNAMENT_SIZE)
    multiple_sim = SimulationRunner(simulation=simulation, results_filename=RESULTS_FILENAME, iterations=ITERATIONS)
    multiple_sim.run_simulation()


if __name__ == "__main__":
    main()
