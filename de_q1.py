import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import fitness_q1

def gen_individual(x_min, x_max, feature_num, rng):
    return np.array([x_min + rng.random()*(x_max-x_min) for _ in range(feature_num)])

def DE(fitness_func, rng, feature_num, pop_size=50, max_iter=3000, max_convergence_iter = 20, x_min=0.0, x_max=1.0, scaling_f=0.5, crossover_rate=0.1):
    pop = [gen_individual(x_min, x_max, feature_num, rng) for _ in range(pop_size)]
    
    best_avg = [] #average fitness of num_best individuals for each generation
    best_individual = pop[0] #get random individual for initial best

    num_iter = 0
    convergence_iter = 0
    prev_best_avg = -1.0 
    while((num_iter < max_iter) & (convergence_iter < max_convergence_iter)):#while not reached stopping criteria
        #mutate
        indices = rng.choice(np.arange(len(pop)), size=5, replace=False)
        to_change = sorted(pop, key= fitness_func)[0]
        #to_change = pop[indices[0]]
        mutated = to_change + scaling_f*(pop[indices[1]] - pop[indices[2]]) + scaling_f*(pop[indices[3]] - pop[indices[4]])
        #crossover
        will_change = rng.integers(low=0, high=feature_num)
        offspring = np.array([ mutated[index] if((rng.random() < crossover_rate) | (index == will_change)) else feature for index, feature in enumerate(to_change)])
        #selection
        if(fitness_func(offspring) < fitness_func(to_change)):
            pop[indices[0]] = offspring
        num_iter += 1

        #calc best average from top 5 individuals and check for convergence
        current_best_avg = np.average([fitness_func(x) for x in pop[:6]])
        best_avg.append(current_best_avg)
        if(np.abs(current_best_avg - prev_best_avg) < 0.0000000000001):
            convergence_iter += 1
        else: convergence_iter = 0
        #get best individual
        best_individual = pop[0]

    return best_individual, fitness_func(best_individual), best_avg, num_iter

def run_de(hyperparameters, D, seeds, fitness_functions, fitness_functions_names):
    for i,fitness_function in enumerate(fitness_functions):
        print('At function ',fitness_functions_names[i])
        hyperparameter = hyperparameters[i]
        best_fitnesses = [] #store all best fitnesses from all the seeds
        iterations = []

        for seed in seeds:
            print('Seed = ', seed)
            rng = np.random.default_rng(seed=seed)
            best, best_fit, best_avg, iter = DE(
                fitness_func=fitness_function,
                rng=rng, pop_size=50, 
                feature_num=D, 
                x_min=-30.0, 
                x_max=30.0, 
                scaling_f=hyperparameter[0], 
                crossover_rate=hyperparameter[1]
                )
            best_fitnesses.append(best_fit)
            iterations.append(iter)
        print('Mean = ', np.average(best_fitnesses), ' Standard Deviation = ', np.std(best_fitnesses),' Average Iterations Taken = ',np.average(iterations))

if __name__ == "__main__":
    D = 20
    print('D = 20')
    seeds = np.random.default_rng(seed=5).integers(low=1,high=200,size=30)
    fitness_functions = [fitness_q1.rosenbrock , fitness_q1.griewanks]
    fitness_functions_names = ['Rosenbrock','Griewanks']

    scaling_f=0.1
    crossover_rate=0.1
    hyperparameters = [(scaling_f, crossover_rate)]
    scaling_f=0.1
    crossover_rate=0.1
    hyperparameters.append((scaling_f, crossover_rate))

    run_de(hyperparameters, D, seeds, fitness_functions, fitness_functions_names)

    D = 50 
    print('D = 50')
    fitness_functions = fitness_functions[:1]
    fitness_functions_names = fitness_functions_names[:1]
    run_de(hyperparameters, D, seeds, fitness_functions, fitness_functions_names)