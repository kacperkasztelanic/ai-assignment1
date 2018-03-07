from SimulationOld import Simulation
from timing import timing

@timing
def main():
    sim = Simulation('had20')
    res = sim.run_simulation()
    no_iter=10
    res = list(map(lambda x: x[:-1], res))
    for i in range(no_iter - 1):
        iteration = sim.run_simulation()
        iteration = list(map(lambda x: x[:-1], iteration))
        res= list(map(lambda x,y: list(map(lambda z, w: z + w, x, y)), iteration, res))
    res = list(map(lambda x: list(map(lambda y: y / no_iter, x)), res))

    print(res)

    min_res = [item[0] for item in res]
    for item in min_res:
        print(str(item) + ' ' + str(round(((item-sim.optimal_cost)/sim.optimal_cost)*100, 3)))

if __name__ == '__main__':
    main()