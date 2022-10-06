import numpy as np

import fitness_q1

def gen_individual(x_min, x_max, feature_num, rng):
    return np.array([x_min + rng.random()*(x_max-x_min) for _ in range(feature_num)])

def DE(fitness_func, rng, feature_num, pop_size=50, max_iter=1000, x_min=0.0, x_max=1.0, scaling_f = 0.5):
    pop = [gen_individual(x_min, x_max, feature_num, rng) for _ in range(pop_size)]
    
    #while not reached stopping criteria
    #mutate
    indices = rng.choice(np.arange(len(pop)), size=3, replace=False)
    mutated = pop[indices[0]] + scaling_f*(pop[indices[1]] - pop[indices[2]])
    #crossover
    offspring = []
    #selection
    #greedy selection: select best 

def run_de(D, seeds, fitness_functions, fitness_functions_names):
    for i,fitness_function in enumerate(fitness_functions):
        print('At function ',fitness_functions_names[i])
        best_fitnesses = [] #store all best fitnesses from all the seeds

        for j,seed in enumerate(seeds):
            print('\tseed = ',seed)

if __name__ == "__main__":
    D = 20
    print('D = 20')
    seeds = np.random.default_rng(seed=5).integers(low=0,high=200,size=3)
    fitness_functions = [fitness_q1.rosenbrock , fitness_q1.griewanks]
    fitness_functions_names = ['Rosenbrock','Griewanks']
    
    rng = np.random.default_rng()
    DE(fitness_func=None, rng=rng, pop_size=5, feature_num=2, x_min=-30.0, x_max=30.0)

    D = 50 
    print('D = 50')
    fitness_functions = fitness_functions[1:]
    fitness_functions_names = fitness_functions_names[1:]