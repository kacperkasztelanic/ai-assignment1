from genetic_solver.SelectionType import SelectionType
from genetic_solver.Simulation import Simulation
from genetic_solver.SimulationRunner import SimulationRunner
from utils.timing import timing
from utils.DataLoader import DataLoader
from utils.DataSaver import DataSaver

SOURCE_DIR = 'data'
SOURCE_EXT = 'dat'
SOLUTION_DIR = 'solutions'
SOLUTION_EXT = 'sln'
RESULTS_DIR = 'results'
RESULTS_EXT = 'csv'

RESULTS_FILENAME = 'results'

FILES = ['had12', 'had14', 'had16', 'had18', 'had20']
FILE_INDEX = 2
ITERATIONS = 10
POPULATION_SIZE = 100
GENERATIONS = 200
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.04
SELECTION_TYPE = SelectionType.TOURNAMENT
TOURNAMENT_SIZE = 5
DIVISION_POINT_RATIO = 0.5


@timing
def main():
    loader = DataLoader(source_dir=SOURCE_DIR, source_ext=SOURCE_EXT, solution_dir=SOLUTION_DIR, solution_ext=SOLUTION_EXT)
    n, flow_matrix, distance_matrix = loader.load_source(FILES[FILE_INDEX])

    simulation = Simulation(n=n, flow_matrix=flow_matrix, distance_matrix=distance_matrix, population_size=POPULATION_SIZE,
                            generations=GENERATIONS, crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB,
                            division_point_ratio=DIVISION_POINT_RATIO, selection_type=SELECTION_TYPE,
                            tournament_size=TOURNAMENT_SIZE)
    multiple_sim = SimulationRunner(simulation=simulation, iterations=ITERATIONS)
    results = multiple_sim.run_simulation()
    
    saver = DataSaver(results_dir=RESULTS_DIR, results_ext=RESULTS_EXT)
    saver.save(result=results, filename=RESULTS_FILENAME)


if __name__ == "__main__":
    main()
