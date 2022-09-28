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
    fitness_func, value_func, feature_num, rng, pop_size = 10, max_iter = 100,
    mut_rate = 0.02, mut_shift = 0.05, num_best = 5, num_worst = 5, max_p = 0.9, min_p = 0.1, learning_rate=0.1):
    p = np.linspace(start=0.5, stop=0.5, num=feature_num) #initialise prob vector

    best_individual = gen_individual(p, feature_num, rng) #initial random guess
    best_avg = [] #average fitness of num_best individuals for each generation

    for i in range(max_iter):  #while not reached stopping criteria
        pop = [gen_individual(p, feature_num, rng) for _ in range(pop_size)] #use p vector to gen pop
        pop.sort(key=fitness_func, reverse=True)  #sort individuals by fitness

        best_individuals = pop[:num_best]
        worst_individuals = pop[pop_size - num_worst:]

        #check if update function is correct
        p = update_prob(p, learning_rate, best_individuals)   #for best individuals
        p = update_prob(p, (-1.0 * learning_rate), worst_individuals)   #for worst individuals

        #p = mutate_prob(p, mut_rate=mut_rate, mut_shift=mut_shift, rng=rng)   #mutate p
        p = np.array([np.maximum(np.minimum(p_i, max_p),min_p) for p_i in p])   #clamp p

        best_avg.append(np.average([value_func(best) for best in best_individuals]))
        best_individual = pop[0]
    
    return best_individual, value_func(best_individual), best_avg, max_iter

"""
Fitness for knapsack problem which focus on finding highest value with weight constraint satisfied
"""
def fitness_function(individual, dataset, penalty_coeff, max_weight):
    if(not individual.__contains__(1)): return 0.0    #to avoid empty knapsack situation
    values, weights = zip(*[ dataset.iloc[i] for i, pickup in enumerate(individual) if (pickup == 1)])
    return np.sum(values) - penalty_coeff*np.max([0.0, np.sum(weights) - capacity])

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

    fit_func = lambda ind : np.sum(ind)
    seeds = np.random.default_rng(seed=50).integers(low=0, high=2000, size=5)

    alpha = 1.0 #penalty coefficient

    for i, (item_num, capacity, dataset) in enumerate(datasets):
        fig, axis = plt.subplots(1, len(seeds))
        fig.set_figwidth(20)
        fig.suptitle(dataset_names[i]+' Convergence Curve')

        best_fitnesses = [] #store all best fitnesses from all the seeds

        print('at dataset',dataset_names[i])
        for j, seed in enumerate(seeds):
            print('\tfor seed',seed)
            best_ind, best_fitness, best_avg, num_iter = PBIL(
                fitness_func=lambda x : fitness_function(individual=x, dataset=dataset, penalty_coeff=alpha, max_weight=capacity), 
                value_func=lambda x : value_fitness(individual=x, dataset=dataset),
                feature_num=item_num, 
                rng=np.random.default_rng(seed=seed)
                )
            print('best individual = ', best_ind,' fitness = ',best_fitness)
            best_fitnesses.append(best_fitness)
            #create convergence curve graph for PBIL output
            sns.lineplot(x=range(num_iter),y=best_avg, ax=axis[j])
            axis[j].set_title('Seed = '+str(seed))
            axis[j].set(xlabel='Number of Generation', ylabel='Average Fitness of Best')

        print('Mean = ', np.average(best_fitnesses), ' Standard Deviation = ', np.std(best_fitnesses))
        fig.savefig('knapsack_'+dataset_names[i]+'.png')
