from multiprocessing import current_process
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import sys

"""
Parse files into tuple (num of items, bag capacity, dataframe)
"""
def parse_data(filepath):
    df = pd.read_table(filepath, sep=' ')
    M = int(df.columns[0])
    Q = int(df.columns[1])
    df.columns.values[0], df.columns.values[1] = 'value','weight'
    return (M,Q,df)  

"""
Generate one individual for pop
    p = probability vector
    feature_num = length of individual
"""
def gen_individual(p, feature_num, rng):
    return np.array([1 if (rng.random() < p[k]) else 0 for k in range(feature_num)])

"""
Update probability vector using selected individuals
    p = probability vector
    learning_rate = +learning_rate if best, -learning_rate if worst
    individuals = best or worst individuals
"""
def update_prob(p, learning_rate, individuals):
    for ind in individuals:
        p = p + learning_rate * (ind - p)
    return p

"""
Probability mutation operator for PBIL
    p = probability vector
    mut_rate = probability of mutation occuring (0,1)
    mut_shift = amount of mutation to affect the prob vector (0,1)
"""
def mutate_prob(p, mut_rate, mut_shift, rng):
    return np.array([
        p_i * (1.0-mut_shift) + rng.random() * mut_shift if(rng.random() < mut_rate) else p_i
        for p_i in p
        ])

"""
The PBIL algorithm 
Returns several values: best solution, best solution's fitness, average of best individuals, number of iterations done
    fitness_func = function to use to measure individual's fitness
    value_func = not used for PBIL fitness, used for graphs and output
    feature_num = length of individual
    pop_size = number of individuals in one population
    max_iter = stopping criteria
    mut_rate = probability of mutation occuring (0,1)
    mut_shift = amount of mutation to affect the prob vector (0,1)
    num_best = number of best individuals
    num_worst =  number of worst individuals
    max_p = maximum value allowed in prob vector
    min_p = minimum value allowed in prob vector
    learning_rate = used for prob vector
"""

def PBIL(
    fitness_func, value_func, feature_num, rng, pop_size = 100, max_iter = 1000, max_convergence_iter = 20,
    mut_rate = 0.02, mut_shift = 0.05, num_best = 5, num_worst = 5, max_p = 0.9, min_p = 0.1, learning_rate=0.1):
    p = np.linspace(start=0.5, stop=0.5, num=feature_num) #initialise prob vector

    best_individual = gen_individual(p, feature_num, rng) #initial random guess
    best_avg = [] #average fitness of num_best individuals for each generation

    iter_num = 0
    convergence_iter = 0
    prev_best_avg = -1.0
    #for i in range(max_iter):  
    while((iter_num < max_iter) & (convergence_iter < max_convergence_iter)):   #while not reached stopping criteria
        pop = [gen_individual(p, feature_num, rng) for _ in range(pop_size)] #use p vector to gen pop
        pop.sort(key=fitness_func, reverse=True)  #sort individuals by fitness

        best_individuals = pop[:num_best]
        worst_individuals = pop[pop_size - num_worst:]

        #calculate best average and check if converging currently
        current_best_avg = np.average([value_func(best) for best in best_individuals])
        if(np.abs(current_best_avg - prev_best_avg) < 0.0000000000001):
            convergence_iter += 1
        prev_best_avg = current_best_avg
        best_avg.append(current_best_avg)
        best_individual = pop[0]

        #update p
        p = update_prob(p, learning_rate, best_individuals)   #for best individuals
        p = update_prob(p, (-1.0 * learning_rate), worst_individuals)   #for worst individuals
        p = mutate_prob(p, mut_rate=mut_rate, mut_shift=mut_shift, rng=rng)   #mutate p
        p = np.array([np.maximum(np.minimum(p_i, max_p),min_p) for p_i in p])   #clamp p

        iter_num += 1
    
    return best_individual, value_func(best_individual), best_avg, iter_num

"""
Fitness for knapsack problem which focus on finding highest value with weight constraint satisfied
"""
def fitness_function(individual, dataset, penalty_coeff, max_weight):
    if(not individual.__contains__(1)): return 0.0    #to avoid empty knapsack situation
    values, weights = zip(*[ dataset.iloc[i] for i, pickup in enumerate(individual) if (pickup == 1)])
    return np.sum(values) - penalty_coeff*np.max([0.0, np.sum(weights) - capacity])
    #return np.sum(values) - penalty_coeff*(np.sum(weights) - capacity)

"""
Evaluates total sum of values in individual while ignoring weight constraint
Not used for PBIL fitness, used for graphs and output
"""
def value_fitness(individual, dataset):
    if(not individual.__contains__(1)): return 0.0    #to avoid empty knapsack situation
    values = [ dataset.iloc[i,0] for i, pickup in enumerate(individual) if (pickup == 1)]
    return np.sum(values)

if __name__ == "__main__":
    filepaths = sys.argv[1:]
    #store dataset as (num of items, bag capacity, data)
    datasets = [parse_data(filepath=filepath) for filepath in filepaths] #parse files
    dataset_names = ['10_269','23_10000','100_995']
    #dataset parameters stored as (alpha, popsize, mutation_shift, max_p, min_p, n_best, n_worst)
    dataset_parameters = [
        (3.0, 100, 0.05, 0.98, 0.02, 2, 2), #10_269
        (3.0, 100, 0.05, 0.95, 0.05, 5, 5), #23_10000
        (10.0, 200, 0.01, 0.99, 0.05, 2, 2)  #100_995
    ]

    fit_func = lambda ind : np.sum(ind)
    seeds = np.random.default_rng(seed=50).integers(low=0, high=2000, size=5)

    datasets = datasets[2:3]
    #dataset_parameters = dataset_parameters[2:3]
    dataset_names = dataset_names[2:3]

    f = open('output.txt', 'w')

    dataset_parameters = [
        (10.0, 200, 0.02, 0.99, 0.01, 2, 2),
        (10.0, 200, 0.02, 0.99, 0.01, 5, 5),
        (10.0, 200, 0.02, 0.99, 0.01, 8, 8),
        (10.0, 200, 0.02, 0.95, 0.05, 2, 2),
        (10.0, 200, 0.02, 0.95, 0.05, 5, 5),
        (10.0, 200, 0.02, 0.95, 0.05, 8, 8),
        (10.0, 200, 0.02, 0.90, 0.1, 2, 2),
        (10.0, 200, 0.02, 0.90, 0.1, 5, 5),
        (10.0, 200, 0.02, 0.90, 0.1, 8, 8)
    ]

    for k in dataset_parameters:
        print('at parameters'+str(k))
        f.write('\n\nat parameters '+str(k)+'\n\n')
        current_parameters = k
        for i, (item_num, capacity, dataset) in enumerate(datasets):
        #current_parameters = dataset_parameters[i]

        #fig, axis = plt.subplots(1, len(seeds))
        #fig.set_figwidth(20)
        #fig.suptitle(dataset_names[i]+' Convergence Curve')

            best_fitnesses = [] #store all best fitnesses from all the seeds

            print('at dataset',dataset_names[i])
            f.write('at dataset'+dataset_names[i]+'\n')
            for j, seed in enumerate(seeds):
                print('seed = ',seed)
                f.write('\tfor seed'+str(seed)+'\n')
                best_ind, best_fitness, best_avg, num_iter = PBIL(
                fitness_func=lambda x : fitness_function(individual=x, dataset=dataset, penalty_coeff=current_parameters[0], max_weight=capacity), 
                value_func=lambda x : value_fitness(individual=x, dataset=dataset),
                feature_num=item_num, 
                rng=np.random.default_rng(seed=seed),
                pop_size=current_parameters[1],
                mut_shift=current_parameters[2],
                max_p= current_parameters[3],
                min_p= current_parameters[4],
                num_best=current_parameters[5],
                num_worst=current_parameters[6]
                )
                f.write('best individual = '+str(best_ind)+' fitness = '+str(best_fitness)+'\n')
                best_fitnesses.append(best_fitness)
                #create convergence curve graph for PBIL output
            #sns.lineplot(x=range(num_iter),y=best_avg, ax=axis[j])
            #axis[j].set_title('Seed = '+str(seed))
            #axis[j].set(xlabel='Number of Generation', ylabel='Average Fitness of Best')

            f.write('Mean = '+str(np.average(best_fitnesses))+ ' Standard Deviation = '+str(np.std(best_fitnesses))+'\n')
        #fig.savefig('knapsack_'+dataset_names[i]+'.png')
