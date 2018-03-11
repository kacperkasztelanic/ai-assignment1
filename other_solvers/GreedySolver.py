import numpy as np


class GreedySolver:
    def __init__(self, n, flow_matrix, distance_matrix):
        self.n = n
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix

    def run(self):
        runs = [self.greedy_with_start_location(i) for i in range(self.n)]
        best_solution, cost = min(runs, key=lambda item: item[1])
        return cost

    def greedy_with_start_location(self, start_location):
        current_location = start_location
        current_factory = 0
        not_visited_locations = list(range(self.n))
        not_visited_factories = list(range(self.n))

        solution = np.zeros(self.n, dtype=np.int)
        solution[current_factory] = current_location

        for _ in range(self.n - 1):
            not_visited_locations.remove(current_location)
            not_visited_factories.remove(current_factory)

            distance = self.distance_matrix[current_factory].reshape((self.n, 1))
            flow = self.distance_matrix[current_location].reshape((1, self.n)).astype(np.float)
            # noinspection PyArgumentList
            routes_matrix = distance @ np.divide(1, flow, where=flow != 0, out=np.zeros_like(flow))

            best_route = np.min(routes_matrix[not_visited_factories][:, not_visited_locations])
            current_factory, current_location = np.where(routes_matrix == best_route)
            if len(current_factory) > 1:
                for factory, location in zip(current_factory, current_location):
                    if factory in not_visited_factories and location in not_visited_locations:
                        current_factory, current_location = factory, location
                        break
            else:
                current_factory, current_location = current_factory[0], current_location[0]
            solution[current_factory] = current_location

        cost = self.cost_function(solution)
        return solution, cost

    def cost_function(self, factories):
        new_distance_matrix = self.distance_matrix[:, factories][factories]
        return np.sum(np.multiply(self.flow_matrix, new_distance_matrix))
