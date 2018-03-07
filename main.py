from MultipleSimulations import SimulationRunner
from Population import SelectionType
from Simulation import Simulation
from timing import timing


@timing
def main():
    files = ['had12', 'had14', 'had16', 'had18', 'had20']
    file_index = 0
    results_filename = 'results.csv'
    iterations = 10
    POPULATION_SIZE = 100
    GENERATIONS = 100
    CROSSOVER_PROB = 0.8
    MUTATION_PROB = 0.4
    SELECTION_TYPE = SelectionType.TOURNAMENT
    TOURNAMENT_SIZE = 10
    DIVISION_POINT_RATIO = 0.5
    simulation = Simulation(filename_root=files[file_index], population_size=POPULATION_SIZE, generations=GENERATIONS,
                            crossover_prob=CROSSOVER_PROB,
                            mutation_prob=MUTATION_PROB, division_point_ratio=DIVISION_POINT_RATIO, selection_type=SELECTION_TYPE,
                            tournament_size=TOURNAMENT_SIZE)
    multiple_sim = SimulationRunner(simulation=simulation, results_filename=results_filename, iterations=iterations)
    multiple_sim.run_simulation()


if __name__ == "__main__":
    main()
